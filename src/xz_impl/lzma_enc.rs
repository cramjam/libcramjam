//! LZMA / LZMA2 encoder — a faithful port of liblzma's encoder stack:
//!
//! * `lz_encoder_mf.c`: the hc3 / hc4 hash-chain and bt2 / bt3 / bt4
//!   binary-tree match finders over a cyclic position buffer, with
//!   `depth`-limited searches, `nice_len` early exit and the periodic
//!   `normalize` of stored positions.
//! * `lzma_encoder_optimum_fast.c`: the greedy-with-lookahead parser used
//!   by presets 0-3 (`LZMA_MODE_FAST`).
//! * `lzma_encoder_optimum_normal.c`: the price-driven dynamic-programming
//!   parser (`optimum[]`, `helper1` / `helper2` / `backward`) used by
//!   presets 4-9 (`LZMA_MODE_NORMAL`), including the length / distance /
//!   alignment price tables and their refresh counters.
//! * `lzma_encoder.c` + `lzma2_encoder.c`: symbol encoding, the per-chunk
//!   encode loop with the 2 MiB uncompressed / 64 KiB compressed limits,
//!   and the "store as uncompressed chunk when compression didn't help"
//!   fallback with its state reset.
//!
//! The whole input is in memory, so the match finder indexes the input
//! slice directly (no sliding window / `move_window`); positions stored in
//! the hash are `index + offset` exactly as in C so the normalization and
//! distance checks carry over unchanged.

use std::io;

use super::lzma::{
    get_dist_state, is_literal_state, update_literal, update_long_rep, update_match,
    update_short_rep, ALIGN_BITS, ALIGN_SIZE, DIST_MODEL_END, DIST_MODEL_START, DIST_SLOTS,
    DIST_SLOT_BITS, DIST_STATES, FULL_DISTANCES, LEN_HIGH_BITS, LEN_HIGH_SYMBOLS,
    LEN_LOW_BITS, LEN_LOW_SYMBOLS, LEN_MID_BITS, LEN_MID_SYMBOLS, LEN_SYMBOLS,
    LITERAL_CODER_SIZE, LZMA_LCLP_MAX, LZMA_PB_MAX, MATCH_LEN_MAX, MATCH_LEN_MIN,
    POS_STATES_MAX, REPS, STATES,
};
use super::options::{LzmaOptions, MatchFinder, Mode};
use super::range_coder::{prob_init, Prob, RangeEncoder};

const ALIGN_MASK: u32 = ALIGN_SIZE as u32 - 1;

// =========================================================================
// Prices (price.h / price_table.c)
// =========================================================================

const RC_MOVE_REDUCING_BITS: u32 = 4;
const RC_BIT_PRICE_SHIFT_BITS: u32 = 4;
const RC_BIT_MODEL_TOTAL: u32 = 1 << 11;
const RC_INFINITY_PRICE: u32 = 1 << 30;

static RC_PRICES: [u8; 128] = [
    128, 103, 91, 84, 78, 73, 69, 66, 63, 61, 58, 56, 54, 52, 51, 49, 48, 46, 45, 44, 43, 42, 41,
    40, 39, 38, 37, 36, 35, 34, 34, 33, 32, 31, 31, 30, 29, 29, 28, 28, 27, 26, 26, 25, 25, 24, 24,
    23, 23, 22, 22, 22, 21, 21, 20, 20, 19, 19, 19, 18, 18, 17, 17, 17, 16, 16, 16, 15, 15, 15, 14,
    14, 14, 13, 13, 13, 12, 12, 12, 11, 11, 11, 11, 10, 10, 10, 10, 9, 9, 9, 9, 8, 8, 8, 8, 7, 7,
    7, 7, 6, 6, 6, 6, 5, 5, 5, 5, 5, 4, 4, 4, 4, 3, 3, 3, 3, 3, 2, 2, 2, 2, 2, 2, 1, 1, 1, 1, 1,
];

#[inline(always)]
fn rc_bit_price(prob: Prob, bit: u32) -> u32 {
    RC_PRICES[((prob as u32 ^ (0u32.wrapping_sub(bit) & (RC_BIT_MODEL_TOTAL - 1))) >> RC_MOVE_REDUCING_BITS) as usize] as u32
}

#[inline(always)]
fn rc_bit_0_price(prob: Prob) -> u32 {
    RC_PRICES[(prob as u32 >> RC_MOVE_REDUCING_BITS) as usize] as u32
}

#[inline(always)]
fn rc_bit_1_price(prob: Prob) -> u32 {
    RC_PRICES[((prob as u32 ^ (RC_BIT_MODEL_TOTAL - 1)) >> RC_MOVE_REDUCING_BITS) as usize] as u32
}

#[inline]
fn rc_bittree_price(probs: &[Prob], bit_levels: u32, mut symbol: u32) -> u32 {
    let mut price = 0;
    symbol += 1 << bit_levels;
    loop {
        let bit = symbol & 1;
        symbol >>= 1;
        price += rc_bit_price(probs[symbol as usize], bit);
        if symbol == 1 {
            break;
        }
    }
    price
}

#[inline]
fn rc_bittree_reverse_price(probs: &[Prob], mut bit_levels: u32, mut symbol: u32) -> u32 {
    let mut price = 0;
    let mut model_index = 1usize;
    loop {
        let bit = symbol & 1;
        symbol >>= 1;
        price += rc_bit_price(probs[model_index], bit);
        model_index = (model_index << 1) + bit as usize;
        bit_levels -= 1;
        if bit_levels == 0 {
            break;
        }
    }
    price
}

#[inline(always)]
fn rc_direct_price(bits: u32) -> u32 {
    bits << RC_BIT_PRICE_SHIFT_BITS
}

// =========================================================================
// fastpos.h
// =========================================================================

#[inline(always)]
fn get_dist_slot(dist: u32) -> u32 {
    if dist <= 4 {
        return dist;
    }
    let i = 31 - dist.leading_zeros();
    (i + i) + ((dist >> (i - 1)) & 1)
}

#[inline(always)]
fn get_dist_slot_2(dist: u32) -> u32 {
    get_dist_slot(dist)
}

// =========================================================================
// memcmplen.h
// =========================================================================

/// Length of the common prefix of `buf[a..]` and `buf[b..]`, starting the
/// comparison at `len` and never exceeding `limit`. Caller guarantees
/// `a + limit <= buf.len()` and `b < a`.
#[inline(always)]
fn memcmplen(buf: &[u8], a: usize, b: usize, mut len: u32, limit: u32) -> u32 {
    debug_assert!(len <= limit);
    debug_assert!(a + limit as usize <= buf.len());
    let p = buf.as_ptr();
    unsafe {
        while len + 8 <= limit {
            let x = core::ptr::read_unaligned(p.add(a + len as usize) as *const u64)
                ^ core::ptr::read_unaligned(p.add(b + len as usize) as *const u64);
            if x != 0 {
                return len + (x.trailing_zeros() >> 3);
            }
            len += 8;
        }
        while len < limit && *p.add(a + len as usize) == *p.add(b + len as usize) {
            len += 1;
        }
    }
    len
}

#[inline(always)]
fn not_equal_16(buf: &[u8], a: usize, b: usize) -> bool {
    buf[a] != buf[b] || buf[a + 1] != buf[b + 1]
}

// =========================================================================
// Match finder (lz_encoder.h / lz_encoder_mf.c / lz_encoder_hash.h)
// =========================================================================

#[derive(Clone, Copy, Default)]
pub struct Match {
    pub len: u32,
    pub dist: u32,
}

const HASH_2_SIZE: u32 = 1 << 10;
const HASH_3_SIZE: u32 = 1 << 16;
const HASH_2_MASK: u32 = HASH_2_SIZE - 1;
const HASH_3_MASK: u32 = HASH_3_SIZE - 1;
const FIX_3_HASH_SIZE: u32 = HASH_2_SIZE;
const FIX_4_HASH_SIZE: u32 = HASH_2_SIZE + HASH_3_SIZE;
const EMPTY_HASH_VALUE: u32 = 0;
const MUST_NORMALIZE_POS: u32 = u32::MAX;

/// liblzma hashes through the CRC32 table (`lzma_crc32_table[0]`).
static CRC32_TABLE: [u32; 256] = {
    let mut t = [0u32; 256];
    let mut i = 0;
    while i < 256 {
        let mut c = i as u32;
        let mut k = 0;
        while k < 8 {
            c = if c & 1 != 0 { (c >> 1) ^ 0xEDB88320 } else { c >> 1 };
            k += 1;
        }
        t[i] = c;
        i += 1;
    }
    t
};

pub struct Mf {
    /// The input buffer. A raw view rather than a borrow so the streaming
    /// driver can own, grow and slide the buffer between calls (it must
    /// call [`Mf::set_buf`] after every change); the one-shot paths point
    /// it at their input slice for the whole run.
    buf_ptr: *const u8,
    buf_len: usize,
    /// `buf[read_pos]` is the next byte to run through the match finder.
    read_pos: u32,
    /// Bytes run through the match finder but not yet encoded.
    read_ahead: u32,
    /// Bytes skipped by `move_pending` at the end of the input that must
    /// be re-hashed once more input arrives (liblzma `mf->pending`).
    pending: u32,
    /// `buf.len()`.
    write_pos: u32,
    /// Stored position = index + offset (starts at `cyclic_size`).
    offset: u32,
    hash: Vec<u32>,
    son: Vec<u32>,
    cyclic_pos: u32,
    cyclic_size: u32,
    hash_mask: u32,
    depth: u32,
    nice_len: u32,
    kind: MatchFinder,
}

impl Mf {
    pub fn new(buf: &[u8], dict_size: u32, kind: MatchFinder, nice_len: u32, depth: u32) -> Self {
        let hash_bytes: u32 = match kind {
            MatchFinder::BinaryTree2 => 2,
            MatchFinder::HashChain3 | MatchFinder::BinaryTree3 => 3,
            MatchFinder::HashChain4 | MatchFinder::BinaryTree4 => 4,
        };
        let is_bt = matches!(kind, MatchFinder::BinaryTree2 | MatchFinder::BinaryTree3 | MatchFinder::BinaryTree4);
        let cyclic_size = dict_size + 1;

        let mut hs: u32;
        if hash_bytes == 2 {
            hs = 0xFFFF;
        } else {
            hs = dict_size - 1;
            hs |= hs >> 1;
            hs |= hs >> 2;
            hs |= hs >> 4;
            hs |= hs >> 8;
            hs >>= 1;
            hs |= 0xFFFF;
            if hs > (1 << 24) {
                if hash_bytes == 3 {
                    hs = (1 << 24) - 1;
                } else {
                    hs >>= 1;
                }
            }
        }
        let hash_mask = hs;
        hs += 1;
        if hash_bytes > 2 {
            hs += HASH_2_SIZE;
        }
        if hash_bytes > 3 {
            hs += HASH_3_SIZE;
        }
        let hash_count = hs as usize;
        let sons_count = cyclic_size as usize * if is_bt { 2 } else { 1 };

        let depth = if depth == 0 {
            if is_bt { 16 + nice_len / 2 } else { 4 + nice_len / 4 }
        } else {
            depth
        };

        Self {
            buf_ptr: buf.as_ptr(),
            buf_len: buf.len(),
            read_pos: 0,
            read_ahead: 0,
            pending: 0,
            write_pos: buf.len() as u32,
            offset: cyclic_size,
            hash: vec![EMPTY_HASH_VALUE; hash_count],
            // C leaves `son` uninitialized; zero is the empty marker anyway.
            son: vec![EMPTY_HASH_VALUE; sons_count],
            cyclic_pos: 0,
            cyclic_size,
            hash_mask,
            depth,
            nice_len,
            kind,
        }
    }

    /// The input buffer.
    #[inline(always)]
    fn buf(&self) -> &[u8] {
        // SAFETY: `buf_ptr`/`buf_len` always describe the caller's live
        // buffer (see `set_buf`); the one-shot paths never change it and
        // the streaming driver refreshes it before every call.
        unsafe { core::slice::from_raw_parts(self.buf_ptr, self.buf_len) }
    }

    /// Point the match finder at (the current version of) its input
    /// buffer; `write_pos` becomes its length.
    pub fn set_buf(&mut self, buf: &[u8]) {
        self.buf_ptr = buf.as_ptr();
        self.buf_len = buf.len();
        self.write_pos = buf.len() as u32;
    }

    #[inline(always)]
    pub fn read_pos(&self) -> u32 {
        self.read_pos
    }

    #[inline(always)]
    pub fn read_ahead(&self) -> u32 {
        self.read_ahead
    }

    /// liblzma `move_window`: drop history so that `keep_size_before` bytes
    /// remain before `read_pos`. Returns how many bytes the caller must
    /// remove from the front of its buffer (then call [`set_buf`]). Stored
    /// positions stay valid because `read_pos + offset` is unchanged.
    pub fn move_window(&mut self, keep_size_before: u32) -> usize {
        debug_assert!(self.read_pos > keep_size_before);
        let move_offset = (self.read_pos - keep_size_before) & !15u32;
        self.offset += move_offset;
        self.read_pos -= move_offset;
        move_offset as usize
    }

    /// `fill_window`'s restart after a sync flush: bytes consumed by
    /// `move_pending` are rewound and run through the hash so later data
    /// can match against them. Doesn't touch `read_ahead`.
    pub fn rehash_pending(&mut self, read_limit: u32) {
        if self.pending > 0 && self.read_pos < read_limit {
            let pending = self.pending;
            self.pending = 0;
            debug_assert!(self.read_pos >= pending);
            self.read_pos -= pending;
            match self.kind {
                MatchFinder::HashChain3 => self.hc3_skip(pending),
                MatchFinder::HashChain4 => self.hc4_skip(pending),
                MatchFinder::BinaryTree2 => self.bt2_skip(pending),
                MatchFinder::BinaryTree3 => self.bt3_skip(pending),
                MatchFinder::BinaryTree4 => self.bt4_skip(pending),
            }
        }
    }

    #[inline(always)]
    pub fn avail(&self) -> u32 {
        self.write_pos - self.read_pos
    }

    #[inline(always)]
    pub fn unencoded(&self) -> u32 {
        self.write_pos - self.read_pos + self.read_ahead
    }

    /// Absolute offset of the byte being encoded next.
    #[inline(always)]
    pub fn position(&self) -> u32 {
        self.read_pos - self.read_ahead
    }

    fn normalize(&mut self) {
        let subvalue = MUST_NORMALIZE_POS - self.cyclic_size;
        for h in self.hash.iter_mut() {
            *h = if *h <= subvalue { EMPTY_HASH_VALUE } else { *h - subvalue };
        }
        for s in self.son.iter_mut() {
            *s = if *s <= subvalue { EMPTY_HASH_VALUE } else { *s - subvalue };
        }
        self.offset -= subvalue;
    }

    #[inline(always)]
    fn move_pos(&mut self) {
        self.cyclic_pos += 1;
        if self.cyclic_pos == self.cyclic_size {
            self.cyclic_pos = 0;
        }
        self.read_pos += 1;
        if self.read_pos.wrapping_add(self.offset) == MUST_NORMALIZE_POS {
            self.normalize();
        }
    }

    /// Too little input left to hash: consume the byte without indexing it
    /// (C's `move_pending`). The streaming driver rewinds and re-hashes
    /// these once more input arrives (`rehash_pending`).
    #[inline(always)]
    fn move_pending(&mut self) {
        self.read_pos += 1;
        self.pending += 1;
    }

    /// C's `header()`: `Some(len_limit)` or `None` after `move_pending`.
    #[inline(always)]
    fn len_limit(&mut self, len_min: u32) -> Option<u32> {
        let avail = self.avail();
        if self.nice_len <= avail {
            Some(self.nice_len)
        } else if avail < len_min {
            self.move_pending();
            None
        } else {
            Some(avail)
        }
    }

    /// SAFETY (hash3/hash4): callers check `avail() >= 3 / 4` first, so
    /// `cur + 2 / cur + 3 < write_pos == buf.len()`.
    #[inline(always)]
    fn hash3(&self, cur: usize) -> (u32, u32) {
        debug_assert!(cur + 3 <= self.buf().len());
        let b = self.buf().as_ptr();
        let (b0, b1, b2) = unsafe { (*b.add(cur), *b.add(cur + 1), *b.add(cur + 2)) };
        let temp = CRC32_TABLE[b0 as usize] ^ b1 as u32;
        let h2 = temp & HASH_2_MASK;
        let hv = (temp ^ ((b2 as u32) << 8)) & self.hash_mask;
        (h2, hv)
    }

    #[inline(always)]
    fn hash4(&self, cur: usize) -> (u32, u32, u32) {
        debug_assert!(cur + 4 <= self.buf().len());
        let b = self.buf().as_ptr();
        let (b0, b1, b2, b3) = unsafe { (*b.add(cur), *b.add(cur + 1), *b.add(cur + 2), *b.add(cur + 3)) };
        let temp = CRC32_TABLE[b0 as usize] ^ b1 as u32;
        let h2 = temp & HASH_2_MASK;
        let h3 = (temp ^ ((b2 as u32) << 8)) & HASH_3_MASK;
        let hv = (temp ^ ((b2 as u32) << 8) ^ (CRC32_TABLE[b3 as usize] << 5)) & self.hash_mask;
        (h2, h3, hv)
    }

    /// `lzma_mf_find`: matches for the current byte (sorted by increasing
    /// length, the last one is the longest), then advance. Returns the
    /// longest length (extended past `nice_len` when it hit it).
    pub fn find(&mut self, matches: &mut [Match; MATCH_LEN_MAX as usize + 1]) -> (u32, usize) {
        let count = match self.kind {
            MatchFinder::HashChain3 => self.hc3_find(matches),
            MatchFinder::HashChain4 => self.hc4_find(matches),
            MatchFinder::BinaryTree2 => self.bt2_find(matches),
            MatchFinder::BinaryTree3 => self.bt3_find(matches),
            MatchFinder::BinaryTree4 => self.bt4_find(matches),
        };
        let mut len_best = 0;
        if count > 0 {
            len_best = matches[count - 1].len;
            if len_best == self.nice_len {
                let mut limit = self.avail() + 1;
                if limit > MATCH_LEN_MAX {
                    limit = MATCH_LEN_MAX;
                }
                let p1 = self.read_pos as usize - 1;
                let p2 = p1 - matches[count - 1].dist as usize - 1;
                len_best = memcmplen(self.buf(), p1, p2, len_best, limit);
            }
        }
        self.read_ahead += 1;
        (len_best, count)
    }

    /// `mf_skip`.
    pub fn skip(&mut self, amount: u32) {
        if amount != 0 {
            match self.kind {
                MatchFinder::HashChain3 => self.hc3_skip(amount),
                MatchFinder::HashChain4 => self.hc4_skip(amount),
                MatchFinder::BinaryTree2 => self.bt2_skip(amount),
                MatchFinder::BinaryTree3 => self.bt3_skip(amount),
                MatchFinder::BinaryTree4 => self.bt4_skip(amount),
            }
            self.read_ahead += amount;
        }
    }

    // ---- hash chain ----

    fn hc_find_func(
        &mut self,
        len_limit: u32,
        pos: u32,
        cur: usize,
        mut cur_match: u32,
        matches: &mut [Match; MATCH_LEN_MAX as usize + 1],
        mut count: usize,
        mut len_best: u32,
    ) -> usize {
        let cyclic_pos = self.cyclic_pos;
        let cyclic_size = self.cyclic_size;
        let mut depth = self.depth;
        // SAFETY: son indices are reduced modulo cyclic_size (son.len() ==
        // cyclic_size for hash chains); `delta < cyclic_size <= pos` keeps
        // `pb` inside the buffer, and `len_best < len_limit <= avail` keeps
        // `cur + len_best < buf.len()`.
        let son = self.son.as_mut_ptr();
        let b = self.buf().as_ptr();
        unsafe {
            *son.add(cyclic_pos as usize) = cur_match;
            loop {
                let delta = pos.wrapping_sub(cur_match);
                if depth == 0 || delta >= cyclic_size {
                    return count;
                }
                depth -= 1;
                let pb = cur - delta as usize;
                cur_match = *son.add(cyclic_pos.wrapping_sub(delta).wrapping_add(if delta > cyclic_pos { cyclic_size } else { 0 }) as usize);
                if *b.add(pb + len_best as usize) == *b.add(cur + len_best as usize) && *b.add(pb) == *b.add(cur) {
                    let len = memcmplen(self.buf(), cur, pb, 1, len_limit);
                    if len_best < len {
                        len_best = len;
                        *matches.get_unchecked_mut(count) = Match { len, dist: delta - 1 };
                        count += 1;
                        if len == len_limit {
                            return count;
                        }
                    }
                }
            }
        }
    }

    #[inline(always)]
    fn hc_skip_one(&mut self, cur_match: u32) {
        self.son[self.cyclic_pos as usize] = cur_match;
        self.move_pos();
    }

    #[inline]
    fn hc3_find(&mut self, matches: &mut [Match; MATCH_LEN_MAX as usize + 1]) -> usize {
        let Some(len_limit) = self.len_limit(3) else { return 0 };
        let cur = self.read_pos as usize;
        let pos = self.read_pos + self.offset;
        let (h2, hv) = self.hash3(cur);
        let delta2 = pos.wrapping_sub(self.hash[h2 as usize]);
        let cur_match = self.hash[(FIX_3_HASH_SIZE + hv) as usize];
        self.hash[h2 as usize] = pos;
        self.hash[(FIX_3_HASH_SIZE + hv) as usize] = pos;

        let mut len_best = 2;
        let mut count = 0usize;
        if delta2 < self.cyclic_size && self.buf()[cur - delta2 as usize] == self.buf()[cur] {
            len_best = memcmplen(self.buf(), cur, cur - delta2 as usize, len_best, len_limit);
            matches[0] = Match { len: len_best, dist: delta2 - 1 };
            count = 1;
            if len_best == len_limit {
                self.hc_skip_one(cur_match);
                return 1;
            }
        }
        let n = self.hc_find_func(len_limit, pos, cur, cur_match, matches, count, len_best);
        self.move_pos();
        n
    }

    #[inline]
    fn hc3_skip(&mut self, mut amount: u32) {
        loop {
            if self.avail() < 3 {
                self.move_pending();
            } else {
                let cur = self.read_pos as usize;
                let pos = self.read_pos + self.offset;
                let (h2, hv) = self.hash3(cur);
                let cur_match = self.hash[(FIX_3_HASH_SIZE + hv) as usize];
                self.hash[h2 as usize] = pos;
                self.hash[(FIX_3_HASH_SIZE + hv) as usize] = pos;
                self.hc_skip_one(cur_match);
            }
            amount -= 1;
            if amount == 0 {
                break;
            }
        }
    }

    #[inline]
    fn hc4_find(&mut self, matches: &mut [Match; MATCH_LEN_MAX as usize + 1]) -> usize {
        let Some(len_limit) = self.len_limit(4) else { return 0 };
        let cur = self.read_pos as usize;
        let pos = self.read_pos + self.offset;
        let (h2, h3, hv) = self.hash4(cur);
        let mut delta2 = pos.wrapping_sub(self.hash[h2 as usize]);
        let delta3 = pos.wrapping_sub(self.hash[(FIX_3_HASH_SIZE + h3) as usize]);
        let cur_match = self.hash[(FIX_4_HASH_SIZE + hv) as usize];
        self.hash[h2 as usize] = pos;
        self.hash[(FIX_3_HASH_SIZE + h3) as usize] = pos;
        self.hash[(FIX_4_HASH_SIZE + hv) as usize] = pos;

        let mut len_best = 1;
        let mut count = 0usize;
        if delta2 < self.cyclic_size && self.buf()[cur - delta2 as usize] == self.buf()[cur] {
            len_best = 2;
            matches[0] = Match { len: 2, dist: delta2 - 1 };
            count = 1;
        }
        if delta2 != delta3 && delta3 < self.cyclic_size && self.buf()[cur - delta3 as usize] == self.buf()[cur] {
            len_best = 3;
            matches[count].dist = delta3 - 1;
            count += 1;
            delta2 = delta3;
        }
        if count != 0 {
            len_best = memcmplen(self.buf(), cur, cur - delta2 as usize, len_best, len_limit);
            matches[count - 1].len = len_best;
            if len_best == len_limit {
                self.hc_skip_one(cur_match);
                return count;
            }
        }
        if len_best < 3 {
            len_best = 3;
        }
        let n = self.hc_find_func(len_limit, pos, cur, cur_match, matches, count, len_best);
        self.move_pos();
        n
    }

    #[inline]
    fn hc4_skip(&mut self, mut amount: u32) {
        loop {
            if self.avail() < 4 {
                self.move_pending();
            } else {
                let cur = self.read_pos as usize;
                let pos = self.read_pos + self.offset;
                let (h2, h3, hv) = self.hash4(cur);
                let cur_match = self.hash[(FIX_4_HASH_SIZE + hv) as usize];
                self.hash[h2 as usize] = pos;
                self.hash[(FIX_3_HASH_SIZE + h3) as usize] = pos;
                self.hash[(FIX_4_HASH_SIZE + hv) as usize] = pos;
                self.hc_skip_one(cur_match);
            }
            amount -= 1;
            if amount == 0 {
                break;
            }
        }
    }

    // ---- binary tree ----

    fn bt_find_func(
        &mut self,
        len_limit: u32,
        pos: u32,
        cur: usize,
        mut cur_match: u32,
        matches: &mut [Match; MATCH_LEN_MAX as usize + 1],
        mut count: usize,
        mut len_best: u32,
    ) -> usize {
        let cyclic_pos = self.cyclic_pos;
        let cyclic_size = self.cyclic_size;
        let mut depth = self.depth;
        // SAFETY: as in `hc_find_func`; son.len() == 2 * cyclic_size here and
        // every index is `2 * (x mod cyclic_size) + {0, 1}`. `len < len_limit`
        // whenever `b[.. + len]` is read (a full-length match returns).
        let son = self.son.as_mut_ptr();
        let b = self.buf().as_ptr();
        let mut ptr0 = ((cyclic_pos as usize) << 1) + 1;
        let mut ptr1 = (cyclic_pos as usize) << 1;
        let mut len0 = 0u32;
        let mut len1 = 0u32;
        unsafe {
            loop {
                let delta = pos.wrapping_sub(cur_match);
                if depth == 0 || delta >= cyclic_size {
                    *son.add(ptr0) = EMPTY_HASH_VALUE;
                    *son.add(ptr1) = EMPTY_HASH_VALUE;
                    return count;
                }
                depth -= 1;
                let pair = (cyclic_pos.wrapping_sub(delta).wrapping_add(if delta > cyclic_pos { cyclic_size } else { 0 }) as usize) << 1;
                let pb = cur - delta as usize;
                let mut len = len0.min(len1);
                if *b.add(pb + len as usize) == *b.add(cur + len as usize) {
                    len = memcmplen(self.buf(), cur, pb, len + 1, len_limit);
                    if len_best < len {
                        len_best = len;
                        *matches.get_unchecked_mut(count) = Match { len, dist: delta - 1 };
                        count += 1;
                        if len == len_limit {
                            *son.add(ptr1) = *son.add(pair);
                            *son.add(ptr0) = *son.add(pair + 1);
                            return count;
                        }
                    }
                }
                if *b.add(pb + len as usize) < *b.add(cur + len as usize) {
                    *son.add(ptr1) = cur_match;
                    ptr1 = pair + 1;
                    cur_match = *son.add(ptr1);
                    len1 = len;
                } else {
                    *son.add(ptr0) = cur_match;
                    ptr0 = pair;
                    cur_match = *son.add(ptr0);
                    len0 = len;
                }
            }
        }
    }

    fn bt_skip_func(&mut self, len_limit: u32, pos: u32, cur: usize, mut cur_match: u32) {
        let cyclic_pos = self.cyclic_pos;
        let cyclic_size = self.cyclic_size;
        let mut depth = self.depth;
        // SAFETY: see `bt_find_func`.
        let son = self.son.as_mut_ptr();
        let b = self.buf().as_ptr();
        let mut ptr0 = ((cyclic_pos as usize) << 1) + 1;
        let mut ptr1 = (cyclic_pos as usize) << 1;
        let mut len0 = 0u32;
        let mut len1 = 0u32;
        unsafe {
            loop {
                let delta = pos.wrapping_sub(cur_match);
                if depth == 0 || delta >= cyclic_size {
                    *son.add(ptr0) = EMPTY_HASH_VALUE;
                    *son.add(ptr1) = EMPTY_HASH_VALUE;
                    return;
                }
                depth -= 1;
                let pair = (cyclic_pos.wrapping_sub(delta).wrapping_add(if delta > cyclic_pos { cyclic_size } else { 0 }) as usize) << 1;
                let pb = cur - delta as usize;
                let mut len = len0.min(len1);
                if *b.add(pb + len as usize) == *b.add(cur + len as usize) {
                    len = memcmplen(self.buf(), cur, pb, len + 1, len_limit);
                    if len == len_limit {
                        *son.add(ptr1) = *son.add(pair);
                        *son.add(ptr0) = *son.add(pair + 1);
                        return;
                    }
                }
                if *b.add(pb + len as usize) < *b.add(cur + len as usize) {
                    *son.add(ptr1) = cur_match;
                    ptr1 = pair + 1;
                    cur_match = *son.add(ptr1);
                    len1 = len;
                } else {
                    *son.add(ptr0) = cur_match;
                    ptr0 = pair;
                    cur_match = *son.add(ptr0);
                    len0 = len;
                }
            }
        }
    }

    #[inline(always)]
    fn hash2(&self, cur: usize) -> u32 {
        (self.buf()[cur] as u32) | ((self.buf()[cur + 1] as u32) << 8)
    }

    #[inline]
    fn bt2_find(&mut self, matches: &mut [Match; MATCH_LEN_MAX as usize + 1]) -> usize {
        let Some(len_limit) = self.len_limit(2) else { return 0 };
        let cur = self.read_pos as usize;
        let pos = self.read_pos + self.offset;
        let hv = self.hash2(cur);
        let cur_match = self.hash[hv as usize];
        self.hash[hv as usize] = pos;
        let n = self.bt_find_func(len_limit, pos, cur, cur_match, matches, 0, 1);
        self.move_pos();
        n
    }

    #[inline]
    fn bt2_skip(&mut self, mut amount: u32) {
        loop {
            if let Some(len_limit) = self.len_limit(2) {
                let cur = self.read_pos as usize;
                let pos = self.read_pos + self.offset;
                let hv = self.hash2(cur);
                let cur_match = self.hash[hv as usize];
                self.hash[hv as usize] = pos;
                self.bt_skip_func(len_limit, pos, cur, cur_match);
                self.move_pos();
            }
            amount -= 1;
            if amount == 0 {
                break;
            }
        }
    }

    #[inline]
    fn bt3_find(&mut self, matches: &mut [Match; MATCH_LEN_MAX as usize + 1]) -> usize {
        let Some(len_limit) = self.len_limit(3) else { return 0 };
        let cur = self.read_pos as usize;
        let pos = self.read_pos + self.offset;
        let (h2, hv) = self.hash3(cur);
        let delta2 = pos.wrapping_sub(self.hash[h2 as usize]);
        let cur_match = self.hash[(FIX_3_HASH_SIZE + hv) as usize];
        self.hash[h2 as usize] = pos;
        self.hash[(FIX_3_HASH_SIZE + hv) as usize] = pos;

        let mut len_best = 2;
        let mut count = 0usize;
        if delta2 < self.cyclic_size && self.buf()[cur - delta2 as usize] == self.buf()[cur] {
            len_best = memcmplen(self.buf(), cur, cur - delta2 as usize, len_best, len_limit);
            matches[0] = Match { len: len_best, dist: delta2 - 1 };
            count = 1;
            if len_best == len_limit {
                self.bt_skip_func(len_limit, pos, cur, cur_match);
                self.move_pos();
                return 1;
            }
        }
        let n = self.bt_find_func(len_limit, pos, cur, cur_match, matches, count, len_best);
        self.move_pos();
        n
    }

    #[inline]
    fn bt3_skip(&mut self, mut amount: u32) {
        loop {
            if let Some(len_limit) = self.len_limit(3) {
                let cur = self.read_pos as usize;
                let pos = self.read_pos + self.offset;
                let (h2, hv) = self.hash3(cur);
                let cur_match = self.hash[(FIX_3_HASH_SIZE + hv) as usize];
                self.hash[h2 as usize] = pos;
                self.hash[(FIX_3_HASH_SIZE + hv) as usize] = pos;
                self.bt_skip_func(len_limit, pos, cur, cur_match);
                self.move_pos();
            }
            amount -= 1;
            if amount == 0 {
                break;
            }
        }
    }

    #[inline]
    fn bt4_find(&mut self, matches: &mut [Match; MATCH_LEN_MAX as usize + 1]) -> usize {
        let Some(len_limit) = self.len_limit(4) else { return 0 };
        let cur = self.read_pos as usize;
        let pos = self.read_pos + self.offset;
        let (h2, h3, hv) = self.hash4(cur);
        let mut delta2 = pos.wrapping_sub(self.hash[h2 as usize]);
        let delta3 = pos.wrapping_sub(self.hash[(FIX_3_HASH_SIZE + h3) as usize]);
        let cur_match = self.hash[(FIX_4_HASH_SIZE + hv) as usize];
        self.hash[h2 as usize] = pos;
        self.hash[(FIX_3_HASH_SIZE + h3) as usize] = pos;
        self.hash[(FIX_4_HASH_SIZE + hv) as usize] = pos;

        let mut len_best = 1;
        let mut count = 0usize;
        if delta2 < self.cyclic_size && self.buf()[cur - delta2 as usize] == self.buf()[cur] {
            len_best = 2;
            matches[0] = Match { len: 2, dist: delta2 - 1 };
            count = 1;
        }
        if delta2 != delta3 && delta3 < self.cyclic_size && self.buf()[cur - delta3 as usize] == self.buf()[cur] {
            len_best = 3;
            matches[count].dist = delta3 - 1;
            count += 1;
            delta2 = delta3;
        }
        if count != 0 {
            len_best = memcmplen(self.buf(), cur, cur - delta2 as usize, len_best, len_limit);
            matches[count - 1].len = len_best;
            if len_best == len_limit {
                self.bt_skip_func(len_limit, pos, cur, cur_match);
                self.move_pos();
                return count;
            }
        }
        if len_best < 3 {
            len_best = 3;
        }
        let n = self.bt_find_func(len_limit, pos, cur, cur_match, matches, count, len_best);
        self.move_pos();
        n
    }

    #[inline]
    fn bt4_skip(&mut self, mut amount: u32) {
        loop {
            if let Some(len_limit) = self.len_limit(4) {
                let cur = self.read_pos as usize;
                let pos = self.read_pos + self.offset;
                let (h2, h3, hv) = self.hash4(cur);
                let cur_match = self.hash[(FIX_4_HASH_SIZE + hv) as usize];
                self.hash[h2 as usize] = pos;
                self.hash[(FIX_3_HASH_SIZE + h3) as usize] = pos;
                self.hash[(FIX_4_HASH_SIZE + hv) as usize] = pos;
                self.bt_skip_func(len_limit, pos, cur, cur_match);
                self.move_pos();
            }
            amount -= 1;
            if amount == 0 {
                break;
            }
        }
    }
}

// =========================================================================
// LZMA1 encoder (lzma_encoder.c / lzma_encoder_private.h)
// =========================================================================

const OPTS: usize = 1 << 12;
const LOOP_INPUT_MAX: u32 = OPTS as u32 + 1;
const LZMA2_CHUNK_MAX: usize = 1 << 16;
const LZMA2_UNCOMPRESSED_MAX: u32 = 1 << 21;
const LZMA2_HEADER_MAX: usize = 6;

pub struct LengthEncoder {
    choice: Prob,
    choice2: Prob,
    low: [[Prob; LEN_LOW_SYMBOLS]; POS_STATES_MAX],
    mid: [[Prob; LEN_MID_SYMBOLS]; POS_STATES_MAX],
    high: [Prob; LEN_HIGH_SYMBOLS],
    prices: Vec<[u32; LEN_SYMBOLS]>,
    table_size: u32,
    counters: [u32; POS_STATES_MAX],
}

impl LengthEncoder {
    fn new(table_size: u32) -> Self {
        Self {
            choice: prob_init(),
            choice2: prob_init(),
            low: [[prob_init(); LEN_LOW_SYMBOLS]; POS_STATES_MAX],
            mid: [[prob_init(); LEN_MID_SYMBOLS]; POS_STATES_MAX],
            high: [prob_init(); LEN_HIGH_SYMBOLS],
            prices: vec![[0u32; LEN_SYMBOLS]; POS_STATES_MAX],
            table_size,
            counters: [0; POS_STATES_MAX],
        }
    }

    fn reset(&mut self, num_pos_states: usize, fast_mode: bool) {
        self.choice = prob_init();
        self.choice2 = prob_init();
        for ps in 0..num_pos_states {
            self.low[ps] = [prob_init(); LEN_LOW_SYMBOLS];
            self.mid[ps] = [prob_init(); LEN_MID_SYMBOLS];
        }
        self.high = [prob_init(); LEN_HIGH_SYMBOLS];
        if !fast_mode {
            for ps in 0..num_pos_states {
                self.update_prices(ps);
            }
        }
    }

    fn update_prices(&mut self, pos_state: usize) {
        let table_size = self.table_size;
        self.counters[pos_state] = table_size;
        let a0 = rc_bit_0_price(self.choice);
        let a1 = rc_bit_1_price(self.choice);
        let b0 = a1 + rc_bit_0_price(self.choice2);
        let b1 = a1 + rc_bit_1_price(self.choice2);
        let mut i = 0u32;
        while i < table_size && i < LEN_LOW_SYMBOLS as u32 {
            self.prices[pos_state][i as usize] = a0 + rc_bittree_price(&self.low[pos_state], LEN_LOW_BITS, i);
            i += 1;
        }
        while i < table_size && i < (LEN_LOW_SYMBOLS + LEN_MID_SYMBOLS) as u32 {
            self.prices[pos_state][i as usize] =
                b0 + rc_bittree_price(&self.mid[pos_state], LEN_MID_BITS, i - LEN_LOW_SYMBOLS as u32);
            i += 1;
        }
        while i < table_size {
            self.prices[pos_state][i as usize] = b1
                + rc_bittree_price(&self.high, LEN_HIGH_BITS, i - (LEN_LOW_SYMBOLS + LEN_MID_SYMBOLS) as u32);
            i += 1;
        }
    }

    #[inline(always)]
    fn price(&self, len: u32, pos_state: usize) -> u32 {
        self.prices[pos_state][(len - MATCH_LEN_MIN) as usize]
    }

    fn encode(&mut self, rc: &mut RangeEncoder, pos_state: usize, len: u32, fast_mode: bool) {
        debug_assert!(len <= MATCH_LEN_MAX);
        let mut len = len - MATCH_LEN_MIN;
        if len < LEN_LOW_SYMBOLS as u32 {
            rc.encode_bit(&mut self.choice, 0);
            rc.encode_bittree(&mut self.low[pos_state], LEN_LOW_BITS, len);
        } else {
            rc.encode_bit(&mut self.choice, 1);
            len -= LEN_LOW_SYMBOLS as u32;
            if len < LEN_MID_SYMBOLS as u32 {
                rc.encode_bit(&mut self.choice2, 0);
                rc.encode_bittree(&mut self.mid[pos_state], LEN_MID_BITS, len);
            } else {
                rc.encode_bit(&mut self.choice2, 1);
                len -= LEN_MID_SYMBOLS as u32;
                rc.encode_bittree(&mut self.high, LEN_HIGH_BITS, len);
            }
        }
        if !fast_mode {
            self.counters[pos_state] -= 1;
            if self.counters[pos_state] == 0 {
                self.update_prices(pos_state);
            }
        }
    }
}

#[derive(Clone, Copy, Default)]
struct Optimal {
    state: u32,
    prev_1_is_literal: bool,
    prev_2: bool,
    pos_prev_2: u32,
    back_prev_2: u32,
    price: u32,
    pos_prev: u32,
    back_prev: u32,
    backs: [u32; REPS],
}

#[inline(always)]
fn make_literal(o: &mut Optimal) {
    o.back_prev = u32::MAX;
    o.prev_1_is_literal = false;
}

#[inline(always)]
fn make_short_rep(o: &mut Optimal) {
    o.back_prev = 0;
    o.prev_1_is_literal = false;
}

pub struct Lzma1Encoder {
    rc: RangeEncoder,
    state: u32,
    reps: [u32; REPS],
    matches: [Match; MATCH_LEN_MAX as usize + 1],
    matches_count: usize,
    longest_match_length: u32,
    fast_mode: bool,
    is_initialized: bool,

    pos_mask: u32,
    lc: u32,
    lp_mask: u32,

    literal: Vec<Prob>,
    is_match: [[Prob; POS_STATES_MAX]; STATES],
    is_rep: [Prob; STATES],
    is_rep0: [Prob; STATES],
    is_rep1: [Prob; STATES],
    is_rep2: [Prob; STATES],
    is_rep0_long: [[Prob; POS_STATES_MAX]; STATES],
    dist_slot: [[Prob; DIST_SLOTS]; DIST_STATES as usize],
    dist_special: [Prob; FULL_DISTANCES - DIST_MODEL_END as usize],
    dist_align: [Prob; ALIGN_SIZE],

    match_len_encoder: LengthEncoder,
    rep_len_encoder: LengthEncoder,

    dist_slot_prices: [[u32; DIST_SLOTS]; DIST_STATES as usize],
    dist_prices: [[u32; FULL_DISTANCES]; DIST_STATES as usize],
    dist_table_size: u32,
    match_price_count: u32,
    align_prices: [u32; ALIGN_SIZE],
    align_price_count: u32,

    opts_end_index: u32,
    opts_current_index: u32,
    opts: Vec<Optimal>,
}

impl Lzma1Encoder {
    pub fn new(opts: &LzmaOptions) -> io::Result<Self> {
        if opts.lc > LZMA_LCLP_MAX
            || opts.lp > LZMA_LCLP_MAX
            || opts.lc + opts.lp > LZMA_LCLP_MAX
            || opts.pb > LZMA_PB_MAX
            || opts.nice_len < MATCH_LEN_MIN
            || opts.nice_len > MATCH_LEN_MAX
        {
            return Err(io::Error::new(io::ErrorKind::InvalidInput, "lzma: invalid encoder options"));
        }
        let fast_mode = opts.mode == Mode::Fast;
        let mut log_size = 0u32;
        while (1u64 << log_size) < opts.dict_size as u64 {
            log_size += 1;
        }
        let table_size = opts.nice_len + 1 - MATCH_LEN_MIN;
        let mut e = Self {
            rc: RangeEncoder::new(),
            state: 0,
            reps: [0; REPS],
            matches: [Match::default(); MATCH_LEN_MAX as usize + 1],
            matches_count: 0,
            longest_match_length: 0,
            fast_mode,
            is_initialized: false,
            pos_mask: (1 << opts.pb) - 1,
            lc: opts.lc,
            lp_mask: (1 << opts.lp) - 1,
            literal: vec![prob_init(); (1usize << (opts.lc + opts.lp)) * LITERAL_CODER_SIZE],
            is_match: [[prob_init(); POS_STATES_MAX]; STATES],
            is_rep: [prob_init(); STATES],
            is_rep0: [prob_init(); STATES],
            is_rep1: [prob_init(); STATES],
            is_rep2: [prob_init(); STATES],
            is_rep0_long: [[prob_init(); POS_STATES_MAX]; STATES],
            dist_slot: [[prob_init(); DIST_SLOTS]; DIST_STATES as usize],
            dist_special: [prob_init(); FULL_DISTANCES - DIST_MODEL_END as usize],
            dist_align: [prob_init(); ALIGN_SIZE],
            match_len_encoder: LengthEncoder::new(table_size),
            rep_len_encoder: LengthEncoder::new(table_size),
            dist_slot_prices: [[0; DIST_SLOTS]; DIST_STATES as usize],
            dist_prices: [[0; FULL_DISTANCES]; DIST_STATES as usize],
            dist_table_size: log_size * 2,
            match_price_count: 0,
            align_prices: [0; ALIGN_SIZE],
            align_price_count: 0,
            opts_end_index: 0,
            opts_current_index: 0,
            opts: if fast_mode { Vec::new() } else { vec![Optimal::default(); OPTS] },
        };
        e.reset();
        Ok(e)
    }

    /// `lzma_lzma_encoder_reset`.
    pub fn reset(&mut self) {
        self.rc.reset();
        self.state = 0;
        self.reps = [0; REPS];
        for p in self.literal.iter_mut() {
            *p = prob_init();
        }
        for i in 0..STATES {
            for j in 0..=self.pos_mask as usize {
                self.is_match[i][j] = prob_init();
                self.is_rep0_long[i][j] = prob_init();
            }
            self.is_rep[i] = prob_init();
            self.is_rep0[i] = prob_init();
            self.is_rep1[i] = prob_init();
            self.is_rep2[i] = prob_init();
        }
        self.dist_special = [prob_init(); FULL_DISTANCES - DIST_MODEL_END as usize];
        self.dist_slot = [[prob_init(); DIST_SLOTS]; DIST_STATES as usize];
        self.dist_align = [prob_init(); ALIGN_SIZE];
        let nps = (self.pos_mask + 1) as usize;
        self.match_len_encoder.reset(nps, self.fast_mode);
        self.rep_len_encoder.reset(nps, self.fast_mode);
        self.match_price_count = u32::MAX / 2;
        self.align_price_count = u32::MAX / 2;
        self.opts_end_index = 0;
        self.opts_current_index = 0;
    }

    // ---- literal ----

    #[inline(always)]
    fn subcoder(&self, pos: u32, prev_byte: u8) -> usize {
        ((((pos & self.lp_mask) << self.lc) + ((prev_byte as u32) >> (8 - self.lc))) as usize) * LITERAL_CODER_SIZE
    }

    fn literal_matched(rc: &mut RangeEncoder, sub: &mut [Prob], match_byte: u32, symbol: u32) {
        let mut offset = 0x100u32;
        let mut symbol = symbol + (1 << 8);
        let mut match_byte = match_byte;
        loop {
            match_byte <<= 1;
            let match_bit = match_byte & offset;
            let idx = (offset + match_bit + (symbol >> 8)) as usize;
            let bit = (symbol >> 7) & 1;
            rc.encode_bit(&mut sub[idx], bit);
            symbol <<= 1;
            offset &= !(match_byte ^ symbol);
            if symbol >= (1 << 16) {
                break;
            }
        }
    }

    fn literal(&mut self, buf: &[u8], mf: &Mf, position: u32) {
        let idx = (mf.read_pos - mf.read_ahead) as usize;
        let cur_byte = buf[idx];
        let base = self.subcoder(position, buf[idx - 1]);
        if is_literal_state(self.state) {
            self.rc.encode_bittree(&mut self.literal[base..base + 0x100], 8, cur_byte as u32);
        } else {
            let match_byte = buf[idx - self.reps[0] as usize - 1];
            Self::literal_matched(&mut self.rc, &mut self.literal[base..base + LITERAL_CODER_SIZE], match_byte as u32, cur_byte as u32);
        }
        self.state = update_literal(self.state);
    }

    // ---- match / rep ----

    fn encode_match(&mut self, pos_state: usize, distance: u32, len: u32) {
        self.state = update_match(self.state);
        self.match_len_encoder.encode(&mut self.rc, pos_state, len, self.fast_mode);
        let dist_slot = get_dist_slot(distance);
        let dist_state = get_dist_state(len) as usize;
        self.rc.encode_bittree(&mut self.dist_slot[dist_state], DIST_SLOT_BITS, dist_slot);
        if dist_slot >= DIST_MODEL_START {
            let footer_bits = (dist_slot >> 1) - 1;
            let base = (2 | (dist_slot & 1)) << footer_bits;
            let dist_reduced = distance - base;
            if dist_slot < DIST_MODEL_END {
                // probs[1..] is used by the reverse bittree; index base-slot-1
                // may be -1 in C, which is why the slice starts one early.
                let start = base as isize - dist_slot as isize - 1;
                rc_bittree_reverse_at(&mut self.rc, &mut self.dist_special, start, footer_bits, dist_reduced);
            } else {
                self.rc.encode_direct_bits(dist_reduced >> ALIGN_BITS, footer_bits - ALIGN_BITS);
                self.rc.encode_bittree_reverse(&mut self.dist_align, ALIGN_BITS, dist_reduced & ALIGN_MASK);
                self.align_price_count += 1;
            }
        }
        self.reps[3] = self.reps[2];
        self.reps[2] = self.reps[1];
        self.reps[1] = self.reps[0];
        self.reps[0] = distance;
        self.match_price_count += 1;
    }

    fn rep_match(&mut self, pos_state: usize, rep: u32, len: u32) {
        let s = self.state as usize;
        if rep == 0 {
            self.rc.encode_bit(&mut self.is_rep0[s], 0);
            self.rc.encode_bit(&mut self.is_rep0_long[s][pos_state], (len != 1) as u32);
        } else {
            let distance = self.reps[rep as usize];
            self.rc.encode_bit(&mut self.is_rep0[s], 1);
            if rep == 1 {
                self.rc.encode_bit(&mut self.is_rep1[s], 0);
            } else {
                self.rc.encode_bit(&mut self.is_rep1[s], 1);
                self.rc.encode_bit(&mut self.is_rep2[s], rep - 2);
                if rep == 3 {
                    self.reps[3] = self.reps[2];
                }
                self.reps[2] = self.reps[1];
            }
            self.reps[1] = self.reps[0];
            self.reps[0] = distance;
        }
        if len == 1 {
            self.state = update_short_rep(self.state);
        } else {
            self.rep_len_encoder.encode(&mut self.rc, pos_state, len, self.fast_mode);
            self.state = update_long_rep(self.state);
        }
    }

    fn encode_symbol(&mut self, buf: &[u8], mf: &mut Mf, back: u32, len: u32, position: u32) {
        let pos_state = (position & self.pos_mask) as usize;
        let s = self.state as usize;
        if back == u32::MAX {
            debug_assert_eq!(len, 1);
            self.rc.encode_bit(&mut self.is_match[s][pos_state], 0);
            self.literal(buf, mf, position);
        } else {
            self.rc.encode_bit(&mut self.is_match[s][pos_state], 1);
            if back < REPS as u32 {
                self.rc.encode_bit(&mut self.is_rep[s], 1);
                self.rep_match(pos_state, back, len);
            } else {
                self.rc.encode_bit(&mut self.is_rep[s], 0);
                self.encode_match(pos_state, back - REPS as u32, len);
            }
        }
        debug_assert!(mf.read_ahead >= len);
        mf.read_ahead -= len;
    }

    fn encode_init(&mut self, buf: &[u8], mf: &mut Mf) -> bool {
        debug_assert_eq!(mf.position(), 0);
        if mf.read_pos == mf.write_pos {
            // Empty input: nothing to do.
        } else {
            mf.skip(1);
            mf.read_ahead = 0;
            self.rc.encode_bit(&mut self.is_match[0][0], 0);
            self.rc.encode_bittree(&mut self.literal[0..0x100], 8, buf[0] as u32);
        }
        self.is_initialized = true;
        true
    }

    /// End-of-payload marker (`encode_eopm`): a match of the minimum
    /// length at distance `UINT32_MAX`.
    fn encode_eopm(&mut self, position: u32) {
        let pos_state = (position & self.pos_mask) as usize;
        let s = self.state as usize;
        self.rc.encode_bit(&mut self.is_match[s][pos_state], 1);
        self.rc.encode_bit(&mut self.is_rep[s], 0);
        self.encode_match(pos_state, u32::MAX, MATCH_LEN_MIN);
    }

    /// `lzma_lzma_encode` for a whole LZMA1 stream (the `.lzma` alone
    /// format): no chunk limits; ends with the end-of-payload marker and
    /// the range-coder flush.
    fn encode_stream(&mut self, buf: &[u8], mf: &mut Mf) {
        if !self.is_initialized {
            self.encode_init(buf, mf);
        }
        let mut position = mf.position();
        while !(mf.read_pos >= mf.write_pos && mf.read_ahead == 0) {
            let (back, len) = if self.fast_mode {
                self.optimum_fast(buf, mf)
            } else {
                self.optimum_normal(buf, mf, position)
            };
            self.encode_symbol(buf, mf, back, len, position);
            position = position.wrapping_add(len);
        }
        self.encode_eopm(position);
        self.rc.finish();
    }

    /// `lzma_lzma_encode` for one LZMA2 chunk: encodes symbols into `rc`
    /// until the uncompressed `limit` (absolute position) or the compressed
    /// chunk limit is reached, then flushes the range coder.
    fn encode_chunk(&mut self, buf: &[u8], mf: &mut Mf, limit: u32) {
        if !self.is_initialized {
            self.encode_init(buf, mf);
        }
        let mut position = mf.position();
        loop {
            if mf.read_pos - mf.read_ahead >= limit
                || self.rc.output.len() + self.rc.cache_size as usize + 4 >= LZMA2_CHUNK_MAX - LOOP_INPUT_MAX as usize
            {
                break;
            }
            if mf.read_pos >= mf.write_pos && mf.read_ahead == 0 {
                break;
            }
            let (back, len) = if self.fast_mode {
                self.optimum_fast(buf, mf)
            } else {
                self.optimum_normal(buf, mf, position)
            };
            self.encode_symbol(buf, mf, back, len, position);
            position = position.wrapping_add(len);
        }
        self.rc.finish();
    }

    /// `lzma_lzma_encode` for the streaming LZMA2 driver: like
    /// [`encode_chunk`] but honours the match finder's `read_limit` and, in
    /// RUN mode, returns `false` with the chunk still open (range coder not
    /// flushed) when more input is needed. Returns `true` once the chunk is
    /// complete and the range coder flushed.
    fn encode_chunk_run(&mut self, buf: &[u8], mf: &mut Mf, limit: u32, read_limit: u32, run: bool) -> bool {
        if !self.is_initialized {
            // C's `encode_init`: nothing can be done in RUN mode without a
            // byte below the read limit.
            if run && mf.read_pos >= read_limit {
                return false;
            }
            self.encode_init(buf, mf);
        }
        let mut position = mf.position();
        loop {
            if mf.read_pos - mf.read_ahead >= limit
                || self.rc.output.len() + self.rc.cache_size as usize + 4 >= LZMA2_CHUNK_MAX - LOOP_INPUT_MAX as usize
            {
                break;
            }
            if mf.read_pos >= read_limit {
                if run {
                    return false;
                }
                if mf.read_ahead == 0 {
                    break;
                }
            }
            let (back, len) = if self.fast_mode {
                self.optimum_fast(buf, mf)
            } else {
                self.optimum_normal(buf, mf, position)
            };
            self.encode_symbol(buf, mf, back, len, position);
            position = position.wrapping_add(len);
        }
        self.rc.finish();
        true
    }

    // =====================================================================
    // Optimum fast (lzma_encoder_optimum_fast.c)
    // =====================================================================

    fn optimum_fast(&mut self, buf: &[u8], mf: &mut Mf) -> (u32, u32) {
        #[inline(always)]
        fn change_pair(small_dist: u32, big_dist: u32) -> bool {
            (big_dist >> 7) > small_dist
        }
        let nice_len = mf.nice_len;
        let (mut len_main, mut matches_count) = if mf.read_ahead == 0 {
            mf.find(&mut self.matches)
        } else {
            debug_assert_eq!(mf.read_ahead, 1);
            (self.longest_match_length, self.matches_count)
        };
        let cur = mf.read_pos as usize - 1;
        let buf_avail = (mf.avail() + 1).min(MATCH_LEN_MAX);
        if buf_avail < 2 {
            return (u32::MAX, 1);
        }

        let mut rep_len = 0u32;
        let mut rep_index = 0u32;
        for i in 0..REPS {
            let back = cur - self.reps[i] as usize - 1;
            if not_equal_16(buf, cur, back) {
                continue;
            }
            let len = memcmplen(buf, cur, back, 2, buf_avail);
            if len >= nice_len {
                mf.skip(len - 1);
                return (i as u32, len);
            }
            if len > rep_len {
                rep_index = i as u32;
                rep_len = len;
            }
        }

        if len_main >= nice_len {
            let back = self.matches[matches_count - 1].dist + REPS as u32;
            mf.skip(len_main - 1);
            return (back, len_main);
        }

        let mut back_main = 0u32;
        if len_main >= 2 {
            back_main = self.matches[matches_count - 1].dist;
            while matches_count > 1 && len_main == self.matches[matches_count - 2].len + 1 {
                if !change_pair(self.matches[matches_count - 2].dist, back_main) {
                    break;
                }
                matches_count -= 1;
                len_main = self.matches[matches_count - 1].len;
                back_main = self.matches[matches_count - 1].dist;
            }
            if len_main == 2 && back_main >= 0x80 {
                len_main = 1;
            }
        }

        if rep_len >= 2
            && (rep_len + 1 >= len_main
                || (rep_len + 2 >= len_main && back_main > (1 << 9))
                || (rep_len + 3 >= len_main && back_main > (1 << 15)))
        {
            mf.skip(rep_len - 1);
            return (rep_index, rep_len);
        }

        if len_main < 2 || buf_avail <= 2 {
            return (u32::MAX, 1);
        }

        // Matches for the next byte: a better one there makes this a literal.
        let (lml, mc) = mf.find(&mut self.matches);
        self.longest_match_length = lml;
        self.matches_count = mc;
        if lml >= 2 {
            let new_dist = self.matches[mc - 1].dist;
            if (lml >= len_main && new_dist < back_main)
                || (lml == len_main + 1 && !change_pair(back_main, new_dist))
                || (lml > len_main + 1)
                || (lml + 1 >= len_main && len_main >= 3 && change_pair(new_dist, back_main))
            {
                return (u32::MAX, 1);
            }
        }

        let cur1 = cur + 1;
        let limit = 2.max(len_main - 1) as usize;
        for i in 0..REPS {
            let back = cur1 - self.reps[i] as usize - 1;
            if buf[cur1..cur1 + limit] == buf[back..back + limit] {
                return (u32::MAX, 1);
            }
        }

        mf.skip(len_main - 2);
        (back_main + REPS as u32, len_main)
    }

    // =====================================================================
    // Optimum normal (lzma_encoder_optimum_normal.c)
    // =====================================================================

    fn get_literal_price(&self, pos: u32, prev_byte: u8, match_mode: bool, match_byte: u32, symbol: u32) -> u32 {
        let base = self.subcoder(pos, prev_byte);
        let sub = &self.literal[base..base + LITERAL_CODER_SIZE];
        if !match_mode {
            return rc_bittree_price(&sub[..0x100], 8, symbol);
        }
        let mut price = 0;
        let mut offset = 0x100u32;
        let mut symbol = symbol + (1 << 8);
        let mut match_byte = match_byte;
        loop {
            match_byte <<= 1;
            let match_bit = match_byte & offset;
            let idx = (offset + match_bit + (symbol >> 8)) as usize;
            let bit = (symbol >> 7) & 1;
            price += rc_bit_price(sub[idx], bit);
            symbol <<= 1;
            offset &= !(match_byte ^ symbol);
            if symbol >= (1 << 16) {
                break;
            }
        }
        price
    }

    #[inline(always)]
    fn get_short_rep_price(&self, state: u32, pos_state: usize) -> u32 {
        rc_bit_0_price(self.is_rep0[state as usize]) + rc_bit_0_price(self.is_rep0_long[state as usize][pos_state])
    }

    #[inline(always)]
    fn get_pure_rep_price(&self, rep_index: u32, state: u32, pos_state: usize) -> u32 {
        let s = state as usize;
        if rep_index == 0 {
            rc_bit_0_price(self.is_rep0[s]) + rc_bit_1_price(self.is_rep0_long[s][pos_state])
        } else {
            let mut price = rc_bit_1_price(self.is_rep0[s]);
            if rep_index == 1 {
                price += rc_bit_0_price(self.is_rep1[s]);
            } else {
                price += rc_bit_1_price(self.is_rep1[s]);
                price += rc_bit_price(self.is_rep2[s], rep_index - 2);
            }
            price
        }
    }

    #[inline(always)]
    fn get_rep_price(&self, rep_index: u32, len: u32, state: u32, pos_state: usize) -> u32 {
        self.rep_len_encoder.price(len, pos_state) + self.get_pure_rep_price(rep_index, state, pos_state)
    }

    #[inline(always)]
    fn get_dist_len_price(&self, dist: u32, len: u32, pos_state: usize) -> u32 {
        let dist_state = get_dist_state(len) as usize;
        let price = if (dist as usize) < FULL_DISTANCES {
            self.dist_prices[dist_state][dist as usize]
        } else {
            let slot = get_dist_slot_2(dist);
            self.dist_slot_prices[dist_state][slot as usize] + self.align_prices[(dist & ALIGN_MASK) as usize]
        };
        price + self.match_len_encoder.price(len, pos_state)
    }

    fn fill_dist_prices(&mut self) {
        for dist_state in 0..DIST_STATES as usize {
            for dist_slot in 0..self.dist_table_size as usize {
                self.dist_slot_prices[dist_state][dist_slot] =
                    rc_bittree_price(&self.dist_slot[dist_state], DIST_SLOT_BITS, dist_slot as u32);
            }
            for dist_slot in DIST_MODEL_END as usize..self.dist_table_size as usize {
                self.dist_slot_prices[dist_state][dist_slot] += rc_direct_price(((dist_slot as u32 >> 1) - 1) - ALIGN_BITS);
            }
            for i in 0..DIST_MODEL_START as usize {
                self.dist_prices[dist_state][i] = self.dist_slot_prices[dist_state][i];
            }
        }
        for i in DIST_MODEL_START..FULL_DISTANCES as u32 {
            let dist_slot = get_dist_slot(i);
            let footer_bits = (dist_slot >> 1) - 1;
            let base = (2 | (dist_slot & 1)) << footer_bits;
            let start = base as isize - dist_slot as isize - 1;
            let price = rc_bittree_reverse_price_at(&self.dist_special, start, footer_bits, i - base);
            for dist_state in 0..DIST_STATES as usize {
                self.dist_prices[dist_state][i as usize] = price + self.dist_slot_prices[dist_state][dist_slot as usize];
            }
        }
        self.match_price_count = 0;
    }

    fn fill_align_prices(&mut self) {
        for i in 0..ALIGN_SIZE {
            self.align_prices[i] = rc_bittree_reverse_price(&self.dist_align, ALIGN_BITS, i as u32);
        }
        self.align_price_count = 0;
    }

    fn backward(&mut self, mut cur: u32) -> (u32, u32) {
        self.opts_end_index = cur;
        let mut pos_mem = self.opts[cur as usize].pos_prev;
        let mut back_mem = self.opts[cur as usize].back_prev;
        loop {
            if self.opts[cur as usize].prev_1_is_literal {
                make_literal(&mut self.opts[pos_mem as usize]);
                self.opts[pos_mem as usize].pos_prev = pos_mem - 1;
                if self.opts[cur as usize].prev_2 {
                    let pp2 = self.opts[cur as usize].pos_prev_2;
                    let bp2 = self.opts[cur as usize].back_prev_2;
                    let o = &mut self.opts[pos_mem as usize - 1];
                    o.prev_1_is_literal = false;
                    o.pos_prev = pp2;
                    o.back_prev = bp2;
                }
            }
            let pos_prev = pos_mem;
            let back_cur = back_mem;
            back_mem = self.opts[pos_prev as usize].back_prev;
            pos_mem = self.opts[pos_prev as usize].pos_prev;
            self.opts[pos_prev as usize].back_prev = back_cur;
            self.opts[pos_prev as usize].pos_prev = cur;
            cur = pos_prev;
            if cur == 0 {
                break;
            }
        }
        self.opts_current_index = self.opts[0].pos_prev;
        (self.opts[0].back_prev, self.opts[0].pos_prev)
    }

    /// Returns `Err((back, len))` when the decision is immediate, else
    /// `Ok(len_end)`.
    fn helper1(&mut self, buf: &[u8], mf: &mut Mf, position: u32) -> Result<u32, (u32, u32)> {
        let nice_len = mf.nice_len;
        let (len_main, matches_count) = if mf.read_ahead == 0 {
            mf.find(&mut self.matches)
        } else {
            debug_assert_eq!(mf.read_ahead, 1);
            (self.longest_match_length, self.matches_count)
        };
        let buf_avail = (mf.avail() + 1).min(MATCH_LEN_MAX);
        if buf_avail < 2 {
            return Err((u32::MAX, 1));
        }
        let cur = mf.read_pos as usize - 1;

        let mut rep_lens = [0u32; REPS];
        let mut rep_max_index = 0usize;
        for i in 0..REPS {
            let back = cur - self.reps[i] as usize - 1;
            if not_equal_16(buf, cur, back) {
                rep_lens[i] = 0;
                continue;
            }
            rep_lens[i] = memcmplen(buf, cur, back, 2, buf_avail);
            if rep_lens[i] > rep_lens[rep_max_index] {
                rep_max_index = i;
            }
        }
        if rep_lens[rep_max_index] >= nice_len {
            let len = rep_lens[rep_max_index];
            mf.skip(len - 1);
            return Err((rep_max_index as u32, len));
        }
        if len_main >= nice_len {
            let back = self.matches[matches_count - 1].dist + REPS as u32;
            mf.skip(len_main - 1);
            return Err((back, len_main));
        }

        let current_byte = buf[cur];
        let match_byte = buf[cur - self.reps[0] as usize - 1];
        if len_main < 2 && current_byte != match_byte && rep_lens[rep_max_index] < 2 {
            return Err((u32::MAX, 1));
        }

        self.opts[0].state = self.state;
        let pos_state = (position & self.pos_mask) as usize;
        let s = self.state as usize;

        self.opts[1].price = rc_bit_0_price(self.is_match[s][pos_state])
            + self.get_literal_price(position, buf[cur - 1], !is_literal_state(self.state), match_byte as u32, current_byte as u32);
        make_literal(&mut self.opts[1]);

        let match_price = rc_bit_1_price(self.is_match[s][pos_state]);
        let rep_match_price = match_price + rc_bit_1_price(self.is_rep[s]);

        if match_byte == current_byte {
            let short_rep_price = rep_match_price + self.get_short_rep_price(self.state, pos_state);
            if short_rep_price < self.opts[1].price {
                self.opts[1].price = short_rep_price;
                make_short_rep(&mut self.opts[1]);
            }
        }

        let len_end = len_main.max(rep_lens[rep_max_index]);
        if len_end < 2 {
            return Err((self.opts[1].back_prev, 1));
        }

        self.opts[1].pos_prev = 0;
        self.opts[0].backs = self.reps;

        let mut len = len_end;
        loop {
            self.opts[len as usize].price = RC_INFINITY_PRICE;
            len -= 1;
            if len < 2 {
                break;
            }
        }

        for i in 0..REPS {
            let mut rep_len = rep_lens[i];
            if rep_len < 2 {
                continue;
            }
            let price = rep_match_price + self.get_pure_rep_price(i as u32, self.state, pos_state);
            loop {
                let cur_and_len_price = price + self.rep_len_encoder.price(rep_len, pos_state);
                let o = &mut self.opts[rep_len as usize];
                if cur_and_len_price < o.price {
                    o.price = cur_and_len_price;
                    o.pos_prev = 0;
                    o.back_prev = i as u32;
                    o.prev_1_is_literal = false;
                }
                rep_len -= 1;
                if rep_len < 2 {
                    break;
                }
            }
        }

        let normal_match_price = match_price + rc_bit_0_price(self.is_rep[s]);
        let mut len = if rep_lens[0] >= 2 { rep_lens[0] + 1 } else { 2 };
        if len <= len_main {
            let mut i = 0usize;
            while len > self.matches[i].len {
                i += 1;
            }
            loop {
                let dist = self.matches[i].dist;
                let cur_and_len_price = normal_match_price + self.get_dist_len_price(dist, len, pos_state);
                let o = &mut self.opts[len as usize];
                if cur_and_len_price < o.price {
                    o.price = cur_and_len_price;
                    o.pos_prev = 0;
                    o.back_prev = dist + REPS as u32;
                    o.prev_1_is_literal = false;
                }
                if len == self.matches[i].len {
                    i += 1;
                    if i == matches_count {
                        break;
                    }
                }
                len += 1;
            }
        }
        Ok(len_end)
    }

    #[allow(clippy::too_many_arguments)]
    fn helper2(
        &mut self,
        reps: &mut [u32; REPS],
        buf: &[u8],
        cur_idx: usize,
        mut len_end: u32,
        position: u32,
        cur: u32,
        nice_len: u32,
        buf_avail_full: u32,
    ) -> u32 {
        let mut matches_count = self.matches_count;
        let mut new_len = self.longest_match_length;
        let cu = cur as usize;
        let mut pos_prev = self.opts[cu].pos_prev;
        let mut state: u32;

        if self.opts[cu].prev_1_is_literal {
            pos_prev -= 1;
            if self.opts[cu].prev_2 {
                state = self.opts[self.opts[cu].pos_prev_2 as usize].state;
                if self.opts[cu].back_prev_2 < REPS as u32 {
                    state = update_long_rep(state);
                } else {
                    state = update_match(state);
                }
            } else {
                state = self.opts[pos_prev as usize].state;
            }
            state = update_literal(state);
        } else {
            state = self.opts[pos_prev as usize].state;
        }

        if pos_prev == cur - 1 {
            if self.opts[cu].back_prev == 0 {
                state = update_short_rep(state);
            } else {
                state = update_literal(state);
            }
        } else {
            let pos: u32;
            if self.opts[cu].prev_1_is_literal && self.opts[cu].prev_2 {
                pos_prev = self.opts[cu].pos_prev_2;
                pos = self.opts[cu].back_prev_2;
                state = update_long_rep(state);
            } else {
                pos = self.opts[cu].back_prev;
                if pos < REPS as u32 {
                    state = update_long_rep(state);
                } else {
                    state = update_match(state);
                }
            }
            let backs = self.opts[pos_prev as usize].backs;
            if pos < REPS as u32 {
                reps[0] = backs[pos as usize];
                let mut i = 1usize;
                while i <= pos as usize {
                    reps[i] = backs[i - 1];
                    i += 1;
                }
                while i < REPS {
                    reps[i] = backs[i];
                    i += 1;
                }
            } else {
                reps[0] = pos - REPS as u32;
                for i in 1..REPS {
                    reps[i] = backs[i - 1];
                }
            }
        }

        self.opts[cu].state = state;
        self.opts[cu].backs = *reps;

        let cur_price = self.opts[cu].price;
        let current_byte = buf[cur_idx];
        let match_byte = buf[cur_idx - reps[0] as usize - 1];
        let pos_state = (position & self.pos_mask) as usize;
        let st = state as usize;

        let cur_and_1_price = cur_price
            + rc_bit_0_price(self.is_match[st][pos_state])
            + self.get_literal_price(position, buf[cur_idx - 1], !is_literal_state(state), match_byte as u32, current_byte as u32);

        let mut next_is_literal = false;
        if cur_and_1_price < self.opts[cu + 1].price {
            self.opts[cu + 1].price = cur_and_1_price;
            self.opts[cu + 1].pos_prev = cur;
            make_literal(&mut self.opts[cu + 1]);
            next_is_literal = true;
        }

        let match_price = cur_price + rc_bit_1_price(self.is_match[st][pos_state]);
        let rep_match_price = match_price + rc_bit_1_price(self.is_rep[st]);

        if match_byte == current_byte && !(self.opts[cu + 1].pos_prev < cur && self.opts[cu + 1].back_prev == 0) {
            let short_rep_price = rep_match_price + self.get_short_rep_price(state, pos_state);
            if short_rep_price <= self.opts[cu + 1].price {
                self.opts[cu + 1].price = short_rep_price;
                self.opts[cu + 1].pos_prev = cur;
                make_short_rep(&mut self.opts[cu + 1]);
                next_is_literal = true;
            }
        }

        if buf_avail_full < 2 {
            return len_end;
        }
        let buf_avail = buf_avail_full.min(nice_len);

        if !next_is_literal && match_byte != current_byte {
            // literal + rep0
            let back = cur_idx - reps[0] as usize - 1;
            let limit = buf_avail_full.min(nice_len + 1);
            let len_test = memcmplen(buf, cur_idx, back, 1, limit) - 1;
            if len_test >= 2 {
                let state_2 = update_literal(state);
                let pos_state_next = ((position + 1) & self.pos_mask) as usize;
                let next_rep_match_price = cur_and_1_price
                    + rc_bit_1_price(self.is_match[state_2 as usize][pos_state_next])
                    + rc_bit_1_price(self.is_rep[state_2 as usize]);
                let offset = cur + 1 + len_test;
                while len_end < offset {
                    len_end += 1;
                    self.opts[len_end as usize].price = RC_INFINITY_PRICE;
                }
                let cur_and_len_price = next_rep_match_price + self.get_rep_price(0, len_test, state_2, pos_state_next);
                let o = &mut self.opts[offset as usize];
                if cur_and_len_price < o.price {
                    o.price = cur_and_len_price;
                    o.pos_prev = cur + 1;
                    o.back_prev = 0;
                    o.prev_1_is_literal = true;
                    o.prev_2 = false;
                }
            }
        }

        let mut start_len = 2u32;

        for rep_index in 0..REPS {
            let back = cur_idx - reps[rep_index] as usize - 1;
            if not_equal_16(buf, cur_idx, back) {
                continue;
            }
            let mut len_test = memcmplen(buf, cur_idx, back, 2, buf_avail);
            while len_end < cur + len_test {
                len_end += 1;
                self.opts[len_end as usize].price = RC_INFINITY_PRICE;
            }
            let len_test_temp = len_test;
            let price = rep_match_price + self.get_pure_rep_price(rep_index as u32, state, pos_state);
            loop {
                let cur_and_len_price = price + self.rep_len_encoder.price(len_test, pos_state);
                let o = &mut self.opts[(cur + len_test) as usize];
                if cur_and_len_price < o.price {
                    o.price = cur_and_len_price;
                    o.pos_prev = cur;
                    o.back_prev = rep_index as u32;
                    o.prev_1_is_literal = false;
                }
                len_test -= 1;
                if len_test < 2 {
                    break;
                }
            }
            len_test = len_test_temp;
            if rep_index == 0 {
                start_len = len_test + 1;
            }

            let mut len_test_2 = len_test + 1;
            let limit = buf_avail_full.min(len_test_2 + nice_len);
            if len_test_2 < limit {
                len_test_2 = memcmplen(buf, cur_idx, back, len_test_2, limit);
            }
            len_test_2 -= len_test + 1;
            if len_test_2 >= 2 {
                let mut state_2 = update_long_rep(state);
                let mut pos_state_next = ((position + len_test) & self.pos_mask) as usize;
                let cur_and_len_literal_price = price
                    + self.rep_len_encoder.price(len_test, pos_state)
                    + rc_bit_0_price(self.is_match[state_2 as usize][pos_state_next])
                    + self.get_literal_price(
                        position + len_test,
                        buf[cur_idx + len_test as usize - 1],
                        true,
                        buf[back + len_test as usize] as u32,
                        buf[cur_idx + len_test as usize] as u32,
                    );
                state_2 = update_literal(state_2);
                pos_state_next = ((position + len_test + 1) & self.pos_mask) as usize;
                let next_rep_match_price = cur_and_len_literal_price
                    + rc_bit_1_price(self.is_match[state_2 as usize][pos_state_next])
                    + rc_bit_1_price(self.is_rep[state_2 as usize]);
                let offset = cur + len_test + 1 + len_test_2;
                while len_end < offset {
                    len_end += 1;
                    self.opts[len_end as usize].price = RC_INFINITY_PRICE;
                }
                let cur_and_len_price = next_rep_match_price + self.get_rep_price(0, len_test_2, state_2, pos_state_next);
                let o = &mut self.opts[offset as usize];
                if cur_and_len_price < o.price {
                    o.price = cur_and_len_price;
                    o.pos_prev = cur + len_test + 1;
                    o.back_prev = 0;
                    o.prev_1_is_literal = true;
                    o.prev_2 = true;
                    o.pos_prev_2 = cur;
                    o.back_prev_2 = rep_index as u32;
                }
            }
        }

        if new_len > buf_avail {
            new_len = buf_avail;
            matches_count = 0;
            while new_len > self.matches[matches_count].len {
                matches_count += 1;
            }
            self.matches[matches_count].len = new_len;
            matches_count += 1;
        }

        if new_len >= start_len {
            let normal_match_price = match_price + rc_bit_0_price(self.is_rep[st]);
            while len_end < cur + new_len {
                len_end += 1;
                self.opts[len_end as usize].price = RC_INFINITY_PRICE;
            }
            let mut i = 0usize;
            while start_len > self.matches[i].len {
                i += 1;
            }
            let mut len_test = start_len;
            loop {
                let cur_back = self.matches[i].dist;
                let mut cur_and_len_price = normal_match_price + self.get_dist_len_price(cur_back, len_test, pos_state);
                {
                    let o = &mut self.opts[(cur + len_test) as usize];
                    if cur_and_len_price < o.price {
                        o.price = cur_and_len_price;
                        o.pos_prev = cur;
                        o.back_prev = cur_back + REPS as u32;
                        o.prev_1_is_literal = false;
                    }
                }
                if len_test == self.matches[i].len {
                    // match + literal + rep0
                    let back = cur_idx - cur_back as usize - 1;
                    let mut len_test_2 = len_test + 1;
                    let limit = buf_avail_full.min(len_test_2 + nice_len);
                    if len_test_2 < limit {
                        len_test_2 = memcmplen(buf, cur_idx, back, len_test_2, limit);
                    }
                    len_test_2 -= len_test + 1;
                    if len_test_2 >= 2 {
                        let mut state_2 = update_match(state);
                        let mut pos_state_next = ((position + len_test) & self.pos_mask) as usize;
                        let cur_and_len_literal_price = cur_and_len_price
                            + rc_bit_0_price(self.is_match[state_2 as usize][pos_state_next])
                            + self.get_literal_price(
                                position + len_test,
                                buf[cur_idx + len_test as usize - 1],
                                true,
                                buf[back + len_test as usize] as u32,
                                buf[cur_idx + len_test as usize] as u32,
                            );
                        state_2 = update_literal(state_2);
                        pos_state_next = (pos_state_next + 1) & self.pos_mask as usize;
                        let next_rep_match_price = cur_and_len_literal_price
                            + rc_bit_1_price(self.is_match[state_2 as usize][pos_state_next])
                            + rc_bit_1_price(self.is_rep[state_2 as usize]);
                        let offset = cur + len_test + 1 + len_test_2;
                        while len_end < offset {
                            len_end += 1;
                            self.opts[len_end as usize].price = RC_INFINITY_PRICE;
                        }
                        cur_and_len_price = next_rep_match_price + self.get_rep_price(0, len_test_2, state_2, pos_state_next);
                        let o = &mut self.opts[offset as usize];
                        if cur_and_len_price < o.price {
                            o.price = cur_and_len_price;
                            o.pos_prev = cur + len_test + 1;
                            o.back_prev = 0;
                            o.prev_1_is_literal = true;
                            o.prev_2 = true;
                            o.pos_prev_2 = cur;
                            o.back_prev_2 = cur_back + REPS as u32;
                        }
                    }
                    i += 1;
                    if i == matches_count {
                        break;
                    }
                }
                len_test += 1;
            }
        }
        len_end
    }

    fn optimum_normal(&mut self, buf: &[u8], mf: &mut Mf, position: u32) -> (u32, u32) {
        if self.opts_end_index != self.opts_current_index {
            debug_assert!(mf.read_ahead > 0);
            let ci = self.opts_current_index as usize;
            let len = self.opts[ci].pos_prev - self.opts_current_index;
            let back = self.opts[ci].back_prev;
            self.opts_current_index = self.opts[ci].pos_prev;
            return (back, len);
        }

        if mf.read_ahead == 0 {
            if self.match_price_count >= (1 << 7) {
                self.fill_dist_prices();
            }
            if self.align_price_count >= ALIGN_SIZE as u32 {
                self.fill_align_prices();
            }
        }

        let mut len_end = match self.helper1(buf, mf, position) {
            Err(r) => return r,
            Ok(l) => l,
        };

        let mut reps = self.reps;
        let mut cur = 1u32;
        while cur < len_end {
            debug_assert!((cur as usize) < OPTS);
            let (lml, mc) = mf.find(&mut self.matches);
            self.longest_match_length = lml;
            self.matches_count = mc;
            if lml >= mf.nice_len {
                break;
            }
            let cur_idx = mf.read_pos as usize - 1;
            let baf = (mf.avail() + 1).min(OPTS as u32 - 1 - cur);
            len_end = self.helper2(&mut reps, buf, cur_idx, len_end, position + cur, cur, mf.nice_len, baf);
            cur += 1;
        }
        self.backward(cur)
    }
}

/// `rc_bittree_reverse` over `probs` with a signed base index (the C code
/// passes `dist_special + base - dist_slot - 1`, which may be one before
/// the array; the tree starts at model index 1 so it never dereferences it).
#[inline]
fn rc_bittree_reverse_at(rc: &mut RangeEncoder, probs: &mut [Prob], start: isize, mut bit_count: u32, mut symbol: u32) {
    let mut model_index = 1isize;
    loop {
        let bit = symbol & 1;
        symbol >>= 1;
        rc.encode_bit(&mut probs[(start + model_index) as usize], bit);
        model_index = (model_index << 1) + bit as isize;
        bit_count -= 1;
        if bit_count == 0 {
            break;
        }
    }
}

#[inline]
fn rc_bittree_reverse_price_at(probs: &[Prob], start: isize, mut bit_levels: u32, mut symbol: u32) -> u32 {
    let mut price = 0;
    let mut model_index = 1isize;
    loop {
        let bit = symbol & 1;
        symbol >>= 1;
        price += rc_bit_price(probs[(start + model_index) as usize], bit);
        model_index = (model_index << 1) + bit as isize;
        bit_levels -= 1;
        if bit_levels == 0 {
            break;
        }
    }
    price
}

// =========================================================================
// LZMA2 driver (lzma2_encoder.c)
// =========================================================================

/// Encode `input` as an LZMA2 stream (the payload of an .xz block) with
/// liblzma's chunking rules: an LZMA chunk holds at most 2 MiB uncompressed
/// / 64 KiB compressed; a chunk that didn't shrink is stored uncompressed
/// (and the next LZMA chunk resets the coder state).
pub fn encode_lzma_to_lzma2(input: &[u8], opts: &LzmaOptions, output: &mut Vec<u8>) -> io::Result<()> {
    if input.is_empty() {
        output.push(0x00);
        return Ok(());
    }
    if input.len() as u64 >= (u32::MAX as u64) - (1u64 << 31) {
        return Err(io::Error::new(io::ErrorKind::InvalidInput, "lzma: input too large for a single block"));
    }
    // The dictionary never needs to exceed the input; it only determines
    // the match-finder table sizes.
    let dict_size = opts.dict_size.max(4096).min(input.len().max(4096) as u32);
    let nice_len = opts.nice_len.clamp(MATCH_LEN_MIN, MATCH_LEN_MAX);
    let mut eopts = opts.clone();
    eopts.dict_size = dict_size;
    eopts.nice_len = nice_len;

    let mut enc = Lzma1Encoder::new(&eopts)?;
    let mut mf = Mf::new(input, dict_size, opts.mf, nice_len, opts.depth);

    let mut need_properties = true;
    let mut need_state_reset = false;
    let mut need_dictionary_reset = true;

    while mf.unencoded() != 0 {
        if need_state_reset {
            enc.reset();
        }
        let read_start = mf.read_pos - mf.read_ahead;
        let limit = read_start + (LZMA2_UNCOMPRESSED_MAX - MATCH_LEN_MAX);
        enc.rc.reset();
        enc.encode_chunk(input, &mut mf, limit);
        let uncompressed_size = (mf.read_pos - mf.read_ahead - read_start) as usize;
        let compressed_size = enc.rc.output.len();
        debug_assert!(compressed_size <= LZMA2_CHUNK_MAX);

        if compressed_size >= uncompressed_size {
            // Store uncompressed: everything the match finder ran through.
            let size = uncompressed_size + mf.read_ahead as usize;
            mf.read_ahead = 0;
            debug_assert!(size <= LZMA2_CHUNK_MAX);
            output.push(if need_dictionary_reset { 1 } else { 2 });
            need_dictionary_reset = false;
            output.push(((size - 1) >> 8) as u8);
            output.push(((size - 1) & 0xFF) as u8);
            let start = mf.read_pos as usize - size;
            output.extend_from_slice(&input[start..start + size]);
            need_state_reset = true;
            continue;
        }

        let mut hdr = [0u8; LZMA2_HEADER_MAX];
        let mut pos = 0usize;
        if need_properties {
            hdr[0] = if need_dictionary_reset { 0x80 + (3 << 5) } else { 0x80 + (2 << 5) };
        } else {
            hdr[0] = if need_state_reset { 0x80 + (1 << 5) } else { 0x80 };
        }
        let us = uncompressed_size - 1;
        hdr[pos] += (us >> 16) as u8;
        pos += 1;
        hdr[pos] = ((us >> 8) & 0xFF) as u8;
        pos += 1;
        hdr[pos] = (us & 0xFF) as u8;
        pos += 1;
        let cs = compressed_size - 1;
        hdr[pos] = (cs >> 8) as u8;
        pos += 1;
        hdr[pos] = (cs & 0xFF) as u8;
        pos += 1;
        if need_properties {
            hdr[pos] = ((opts.pb * 5 + opts.lp) * 9 + opts.lc) as u8;
            pos += 1;
        }
        need_properties = false;
        need_state_reset = false;
        need_dictionary_reset = false;
        output.extend_from_slice(&hdr[..pos]);
        output.extend_from_slice(&enc.rc.output);
    }
    output.push(0x00);
    Ok(())
}

/// Encode `input` in the legacy `.lzma` ("alone") format: the 13-byte
/// header (properties byte, dictionary size, unknown uncompressed size) and
/// one LZMA1 stream terminated by the end-of-payload marker — the layout
/// Python's `lzma.compress(format=FORMAT_ALONE)` / liblzma's alone encoder
/// produce.
pub fn encode_lzma_alone(input: &[u8], opts: &LzmaOptions, output: &mut Vec<u8>) -> io::Result<()> {
    let dict_size = opts.dict_size.max(4096);
    let nice_len = opts.nice_len.clamp(MATCH_LEN_MIN, MATCH_LEN_MAX);
    let mut eopts = opts.clone();
    // Match-finder tables only need to cover the input.
    eopts.dict_size = dict_size.min(input.len().max(4096) as u32);
    eopts.nice_len = nice_len;

    output.push(((opts.pb * 5 + opts.lp) * 9 + opts.lc) as u8);
    output.extend_from_slice(&dict_size.to_le_bytes());
    output.extend_from_slice(&u64::MAX.to_le_bytes());

    let mut enc = Lzma1Encoder::new(&eopts)?;
    let mut mf = Mf::new(input, eopts.dict_size, opts.mf, nice_len, opts.depth);
    enc.encode_stream(input, &mut mf);
    output.extend_from_slice(&enc.rc.output);
    Ok(())
}

// =========================================================================
// Streaming LZMA2 encoder (lz_encoder.c + lzma2_encoder.c)
// =========================================================================

/// What the caller wants from [`Lzma2StreamEncoder::encode`] — liblzma's
/// `lzma_action` for the encoder side.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Action {
    /// Keep `keep_size_after` bytes of lookahead unencoded, wait for more.
    Run,
    /// `LZMA_SYNC_FLUSH`: finish the current chunk so everything written so
    /// far is decodable; the dictionary is kept.
    Flush,
    /// `LZMA_FINISH`: like `Flush`, then the LZMA2 end marker.
    Finish,
}

/// Incremental LZMA2 encoder: the match finder works over a sliding window
/// (`keep_size_before` = dict + OPTS + 64 KiB of history, plus a reserve so
/// the memmove is amortised), input is consumed as it arrives, chunks are
/// emitted into [`out`](Self::out) exactly as `lzma2_encode` does. A single
/// `write` + `Finish` produces the same bytes as [`encode_lzma_to_lzma2`]
/// for inputs that fit in one window.
pub struct Lzma2StreamEncoder {
    opts: LzmaOptions,
    enc: Lzma1Encoder,
    mf: Mf,
    buf: Vec<u8>,
    keep_size_before: u32,
    keep_size_after: u32,
    capacity: u32,
    read_limit: u32,
    chunk_open: bool,
    uncompressed_size: u32,
    need_properties: bool,
    need_state_reset: bool,
    need_dictionary_reset: bool,
    finished: bool,
    /// Encoded LZMA2 bytes ready for the caller to drain.
    pub out: Vec<u8>,
}

impl Lzma2StreamEncoder {
    pub fn new(opts: &LzmaOptions) -> io::Result<Self> {
        let dict_size = opts.dict_size.max(4096);
        let nice_len = opts.nice_len.clamp(MATCH_LEN_MIN, MATCH_LEN_MAX);
        let mut eopts = opts.clone();
        eopts.dict_size = dict_size;
        eopts.nice_len = nice_len;
        let enc = Lzma1Encoder::new(&eopts)?;
        let mf = Mf::new(&[], dict_size, opts.mf, nice_len, opts.depth);
        // lz_encoder_prepare(): before_size = OPTS (raised so that
        // before + dict >= LZMA2_CHUNK_MAX for the uncompressed-chunk copy),
        // after_size = LOOP_INPUT_MAX, plus match_len_max.
        let before_size = (OPTS as u32).max((LZMA2_CHUNK_MAX as u32).saturating_sub(dict_size));
        let keep_size_before = before_size + dict_size;
        let keep_size_after = LOOP_INPUT_MAX + MATCH_LEN_MAX;
        let mut reserve = dict_size / 2;
        if reserve > (1u32 << 30) {
            reserve /= 2;
        }
        reserve += (before_size + MATCH_LEN_MAX + LOOP_INPUT_MAX) / 2 + (1u32 << 19);
        let capacity = keep_size_before + reserve + keep_size_after;
        Ok(Self {
            opts: eopts,
            enc,
            mf,
            buf: Vec::with_capacity(1 << 16),
            keep_size_before,
            keep_size_after,
            capacity,
            read_limit: 0,
            chunk_open: false,
            uncompressed_size: 0,
            need_properties: true,
            need_state_reset: false,
            need_dictionary_reset: true,
            finished: false,
            out: Vec::new(),
        })
    }

    /// Bytes fed in but not yet encoded.
    pub fn unencoded(&self) -> u32 {
        self.mf.unencoded()
    }

    /// `fill_window`'s window move: once the encoder has consumed past the
    /// reserve, drop everything older than `keep_size_before`.
    fn maybe_move_window(&mut self) {
        if self.mf.read_pos() >= self.capacity - self.keep_size_after && self.mf.read_pos() > self.keep_size_before {
            let off = self.mf.move_window(self.keep_size_before);
            if off > 0 {
                self.buf.drain(..off);
                self.read_limit = self.read_limit.saturating_sub(off as u32);
                self.mf.set_buf(&self.buf);
            }
        }
    }

    /// Feed input (`LZMA_RUN`): appended to the window in pieces that fit
    /// the buffer, encoding as it goes and keeping `keep_size_after` bytes
    /// of lookahead unencoded.
    pub fn write(&mut self, mut data: &[u8]) {
        debug_assert!(!self.finished);
        while !data.is_empty() {
            self.maybe_move_window();
            let mut room = (self.capacity as usize).saturating_sub(self.buf.len());
            if room == 0 {
                // Buffer full but nothing could be dropped yet: encode what
                // is allowed, then the window can move.
                self.encode(Action::Run);
                self.maybe_move_window();
                room = (self.capacity as usize).saturating_sub(self.buf.len()).max(1);
            }
            let take = room.min(data.len());
            self.buf.extend_from_slice(&data[..take]);
            data = &data[take..];
            self.mf.set_buf(&self.buf);
            let write_pos = self.mf.write_pos;
            if write_pos > self.keep_size_after {
                self.read_limit = write_pos - self.keep_size_after;
            }
            self.mf.rehash_pending(self.read_limit);
            self.encode(Action::Run);
        }
    }

    /// `LZMA_SYNC_FLUSH` / `LZMA_FINISH`: allow the encoder to consume
    /// everything, close the open chunk, and (for `Finish`) write the end
    /// marker. Output lands in [`out`](Self::out).
    pub fn finish_input(&mut self, action: Action) {
        debug_assert!(action != Action::Run);
        self.read_limit = self.mf.write_pos;
        self.mf.rehash_pending(self.read_limit);
        self.encode(action);
    }

    /// `lzma2_encode`.
    fn encode(&mut self, action: Action) {
        loop {
            if !self.chunk_open {
                if self.mf.unencoded() == 0 {
                    if action == Action::Finish && !self.finished {
                        self.out.push(0x00);
                        self.finished = true;
                    }
                    return;
                }
                if self.need_state_reset {
                    self.enc.reset();
                }
                self.enc.rc.reset();
                self.uncompressed_size = 0;
                self.chunk_open = true;
            }

            let left = LZMA2_UNCOMPRESSED_MAX - self.uncompressed_size;
            let limit = if left < MATCH_LEN_MAX { 0 } else { self.mf.position() + left - MATCH_LEN_MAX };
            let read_start = self.mf.position();
            let done = self
                .enc
                .encode_chunk_run(&self.buf, &mut self.mf, limit, self.read_limit, action == Action::Run);
            self.uncompressed_size += self.mf.position() - read_start;
            debug_assert!(self.uncompressed_size <= LZMA2_UNCOMPRESSED_MAX);
            if !done {
                return;
            }

            let compressed_size = self.enc.rc.output.len();
            debug_assert!(compressed_size <= LZMA2_CHUNK_MAX);
            if compressed_size >= self.uncompressed_size as usize {
                // Didn't shrink: store the chunk (everything the match
                // finder ran through) uncompressed; the next LZMA chunk
                // resets the coder state.
                let size = self.uncompressed_size as usize + self.mf.read_ahead() as usize;
                self.mf.read_ahead = 0;
                debug_assert!(size <= LZMA2_CHUNK_MAX);
                self.out.push(if self.need_dictionary_reset { 1 } else { 2 });
                self.need_dictionary_reset = false;
                self.out.push(((size - 1) >> 8) as u8);
                self.out.push(((size - 1) & 0xFF) as u8);
                let end = self.mf.read_pos() as usize;
                self.out.extend_from_slice(&self.buf[end - size..end]);
                self.need_state_reset = true;
            } else {
                let mut hdr = [0u8; LZMA2_HEADER_MAX];
                let mut pos = 0usize;
                if self.need_properties {
                    hdr[0] = if self.need_dictionary_reset { 0x80 + (3 << 5) } else { 0x80 + (2 << 5) };
                } else {
                    hdr[0] = if self.need_state_reset { 0x80 + (1 << 5) } else { 0x80 };
                }
                let us = self.uncompressed_size as usize - 1;
                hdr[pos] += (us >> 16) as u8;
                pos += 1;
                hdr[pos] = ((us >> 8) & 0xFF) as u8;
                pos += 1;
                hdr[pos] = (us & 0xFF) as u8;
                pos += 1;
                let cs = compressed_size - 1;
                hdr[pos] = (cs >> 8) as u8;
                pos += 1;
                hdr[pos] = (cs & 0xFF) as u8;
                pos += 1;
                if self.need_properties {
                    hdr[pos] = ((self.opts.pb * 5 + self.opts.lp) * 9 + self.opts.lc) as u8;
                    pos += 1;
                }
                self.need_properties = false;
                self.need_state_reset = false;
                self.need_dictionary_reset = false;
                self.out.extend_from_slice(&hdr[..pos]);
                self.out.extend_from_slice(&self.enc.rc.output);
            }
            self.chunk_open = false;
        }
    }
}
