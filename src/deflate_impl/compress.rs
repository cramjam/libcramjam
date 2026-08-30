//! DEFLATE compression (RFC 1951) — a port of zlib's `deflate.c`.
//!
//! Levels 1-3 use `deflate_fast` (greedy, short chains), 4-9 `deflate_slow`
//! (lazy evaluation) with zlib's `configuration_table`; `longest_match` is
//! zlib's chain walk with the `prev_length` early-outs. Blocks are flushed
//! every `LIT_BUFSIZE - 1` symbols (memLevel 8) and emitted by `trees.rs`
//! (stored / static / dynamic chosen by computed bit lengths). The whole
//! input is in memory, so there is no sliding window: positions index the
//! input directly and the hash chain is a 32 KiB ring keyed by position.

use super::bitwriter::BitWriter;
use super::trees::{Sym, TreeState};

const MIN_MATCH: usize = 3;
const MAX_MATCH: usize = 258;
/// Maximum back-reference distance (RFC 1951 allows exactly 32 KiB).
const MAX_DIST: usize = 32768;
const W_MASK: usize = MAX_DIST - 1;
const HASH_BITS: u32 = 15; // memLevel 8 → hash_bits = 15
const HASH_SIZE: usize = 1 << HASH_BITS;
const HASH_MASK: u32 = (HASH_SIZE - 1) as u32;
const HASH_SHIFT: u32 = (HASH_BITS + MIN_MATCH as u32 - 1) / MIN_MATCH as u32; // 5
/// zlib `lit_bufsize` at memLevel 8; a block is flushed at `LIT_BUFSIZE - 1` symbols.
const LIT_BUFSIZE: usize = 1 << 14;
const SYM_END: usize = LIT_BUFSIZE - 1;
/// Matches of length 3 farther back than this are dropped in the lazy parser.
const TOO_FAR: usize = 4096;
const MAX_STORED_BLOCK: usize = 65535;
const NIL: u32 = u32::MAX;

/// Level table: good_length, max_lazy, nice_length, max_chain (zlib's
/// `configuration_table` semantics: `max_lazy` is the max insert length for
/// the greedy levels 1-3 and the lazy-evaluation threshold for 4-9).
///
/// Tuned against miniz_oxide's level table (flate2's default backend) so
/// every level is at or below miniz on BOTH bytes and time on the bench
/// subset: miniz L1 is a 1-probe greedy mode, L2-3 greedy with 6/32 probes
/// inserting every position, L4-9 lazy with 16/32/128/256/512/768 probes.
/// zlib's own L8/L9 chains (1024/4096) compress ~0.05pp better but cost
/// 1.3-1.5x the time; L1's `max_lazy = 0` skips in-match inserts.
struct Config {
    good_length: usize,
    max_lazy: usize,
    nice_length: usize,
    max_chain: usize,
}

const CONFIGS: [Config; 10] = [
    Config { good_length: 0, max_lazy: 0, nice_length: 0, max_chain: 0 }, // 0: stored
    Config { good_length: 4, max_lazy: 0, nice_length: 8, max_chain: 1 }, // 1: greedy
    Config { good_length: 4, max_lazy: 258, nice_length: 16, max_chain: 4 }, // 2: greedy
    Config { good_length: 4, max_lazy: 258, nice_length: 32, max_chain: 32 }, // 3: greedy
    Config { good_length: 4, max_lazy: 32, nice_length: 32, max_chain: 16 }, // 4: lazy
    Config { good_length: 8, max_lazy: 16, nice_length: 32, max_chain: 28 }, // 5
    Config { good_length: 8, max_lazy: 16, nice_length: 128, max_chain: 128 }, // 6
    Config { good_length: 8, max_lazy: 32, nice_length: 128, max_chain: 256 }, // 7
    Config { good_length: 32, max_lazy: 128, nice_length: 258, max_chain: 320 }, // 8
    Config { good_length: 16, max_lazy: 258, nice_length: 258, max_chain: 512 }, // 9
];

/// Compress `input` into a raw DEFLATE stream.
pub fn deflate(input: &[u8], level: u32) -> Vec<u8> {
    let level = std::cmp::min(level, 9) as usize;
    if level == 0 || input.is_empty() {
        return compress_stored_all(input);
    }
    let mut d = Deflater::new(input, level);
    if level <= 3 {
        d.deflate_fast();
    } else {
        d.deflate_slow();
    }
    d.w.finish()
}

// ---------------------------------------------------------------------------
// Stored-block compression (level 0)
// ---------------------------------------------------------------------------

fn compress_stored_all(input: &[u8]) -> Vec<u8> {
    let mut w = BitWriter::with_capacity(input.len() + input.len() / MAX_STORED_BLOCK * 5 + 20);
    if input.is_empty() {
        write_stored_block(&mut w, &[], true);
        return w.finish();
    }
    let mut offset = 0;
    while offset < input.len() {
        let chunk = std::cmp::min(MAX_STORED_BLOCK, input.len() - offset);
        let is_final = offset + chunk >= input.len();
        write_stored_block(&mut w, &input[offset..offset + chunk], is_final);
        offset += chunk;
    }
    w.finish()
}

fn write_stored_block(w: &mut BitWriter, data: &[u8], is_final: bool) {
    w.write_bits(is_final as u32, 3); // BFINAL + BTYPE 00
    w.align_to_byte();
    let len = data.len() as u16;
    w.write_u16_le(len);
    w.write_u16_le(!len);
    w.write_bytes(data);
}

// ---------------------------------------------------------------------------
// LZ77 parser (zlib deflate_fast / deflate_slow)
// ---------------------------------------------------------------------------

struct Deflater<'a> {
    input: &'a [u8],
    cfg: &'static Config,
    w: BitWriter,
    trees: TreeState,
    /// Hash chain heads (absolute positions) and a ring of *deltas* to the
    /// previous position with the same hash (0 = end of chain). A u16 ring is
    /// half the size of zlib's absolute-position `prev` and stays in L2.
    head: Vec<u32>,
    prev: Vec<u16>,
    /// Symbol buffer of the current block and the byte where it starts.
    syms: Vec<Sym>,
    block_start: usize,
    strstart: usize,
    /// Result of the last `longest_match`.
    match_start: usize,
    match_length: usize,
    prev_length: usize,
    prev_match: usize,
    match_available: bool,
}

impl<'a> Deflater<'a> {
    fn new(input: &'a [u8], level: usize) -> Self {
        Self {
            input,
            cfg: &CONFIGS[level],
            w: BitWriter::with_capacity(input.len() / 2 + 64),
            trees: TreeState::new(),
            head: vec![NIL; HASH_SIZE],
            prev: vec![0; MAX_DIST],
            syms: Vec::with_capacity(LIT_BUFSIZE),
            block_start: 0,
            strstart: 0,
            match_start: 0,
            match_length: MIN_MATCH - 1,
            prev_length: MIN_MATCH - 1,
            prev_match: 0,
            match_available: false,
        }
    }

    #[inline(always)]
    fn lookahead(&self) -> usize {
        self.input.len() - self.strstart
    }

    /// zlib's rolling hash of the 3 bytes at `pos` (same value as
    /// `UPDATE_HASH` applied byte by byte).
    #[inline(always)]
    fn hash_at(&self, pos: usize) -> u32 {
        // SAFETY: callers guarantee pos + 3 <= input.len().
        unsafe {
            let p = self.input.as_ptr().add(pos);
            ((((*p as u32) << HASH_SHIFT) ^ (*p.add(1) as u32)) << HASH_SHIFT ^ (*p.add(2) as u32)) & HASH_MASK
        }
    }

    /// `INSERT_STRING`: returns the previous head of the chain.
    #[inline(always)]
    fn insert_string(&mut self, pos: usize) -> u32 {
        let h = self.hash_at(pos) as usize;
        // SAFETY: h < HASH_SIZE, pos & W_MASK < MAX_DIST.
        unsafe {
            let head = *self.head.get_unchecked(h);
            // Distance to the previous occurrence; beyond the window (or no
            // previous occurrence) the chain ends here.
            let delta = pos.wrapping_sub(head as usize);
            *self.prev.get_unchecked_mut(pos & W_MASK) = if head == NIL || delta > MAX_DIST { 0 } else { delta as u16 };
            *self.head.get_unchecked_mut(h) = pos as u32;
            head
        }
    }

    /// zlib `longest_match`. Uses `prev_length` as the length to beat.
    #[inline(never)]
    fn longest_match(&mut self, mut cur_match: usize) -> usize {
        let input = self.input;
        let strstart = self.strstart;
        let lookahead = input.len() - strstart;
        let mut chain_length = self.cfg.max_chain;
        let mut best_len = self.prev_length;
        let mut nice_match = self.cfg.nice_length;
        if self.prev_length >= self.cfg.good_length {
            chain_length >>= 2;
        }
        if nice_match > lookahead {
            nice_match = lookahead;
        }
        let max_len = MAX_MATCH.min(lookahead);
        if best_len >= max_len {
            return best_len.min(lookahead);
        }
        let limit = strstart.saturating_sub(MAX_DIST);
        let base = input.as_ptr();
        // SAFETY: every index below is < input.len(): cur_match < strstart,
        // and compares stay below strstart + max_len <= input.len().
        unsafe {
            let scan = base.add(strstart);
            let mut scan_end1 = *scan.add(best_len - 1);
            let mut scan_end = *scan.add(best_len);
            loop {
                let m = base.add(cur_match);
                if *m.add(best_len) == scan_end
                    && *m.add(best_len - 1) == scan_end1
                    && *m == *scan
                    && *m.add(1) == *scan.add(1)
                {
                    // Extend from byte 2 in 8-byte steps.
                    let mut len = 2usize;
                    while len + 8 <= max_len {
                        let a = core::ptr::read_unaligned(scan.add(len) as *const u64);
                        let b = core::ptr::read_unaligned(m.add(len) as *const u64);
                        let x = a ^ b;
                        if x != 0 {
                            len += (x.trailing_zeros() >> 3) as usize;
                            break;
                        }
                        len += 8;
                    }
                    if len + 8 > max_len && len < max_len {
                        // Tail (also reached when the 8-byte loop ended
                        // without a mismatch).
                        while len < max_len && *scan.add(len) == *m.add(len) {
                            len += 1;
                        }
                    }
                    if len > best_len {
                        self.match_start = cur_match;
                        best_len = len;
                        if len >= nice_match || len >= max_len {
                            break;
                        }
                        scan_end1 = *scan.add(best_len - 1);
                        scan_end = *scan.add(best_len);
                    }
                }
                let delta = *self.prev.get_unchecked(cur_match & W_MASK) as usize;
                if delta == 0 || cur_match < limit + delta {
                    break;
                }
                cur_match -= delta;
                chain_length -= 1;
                if chain_length == 0 {
                    break;
                }
            }
        }
        best_len.min(lookahead)
    }

    #[inline(always)]
    fn tally_lit(&mut self, c: u8) -> bool {
        self.syms.push(Sym { dist: 0, lc: c });
        self.trees.dyn_ltree[c as usize].fc += 1;
        self.syms.len() == SYM_END
    }

    #[inline(always)]
    fn tally_dist(&mut self, dist: usize, len: usize) -> bool {
        use super::trees::{d_code, LENGTH_CODE, LITERALS};
        self.syms.push(Sym { dist: dist as u16, lc: len as u8 });
        self.trees.dyn_ltree[LENGTH_CODE[len] as usize + LITERALS + 1].fc += 1;
        self.trees.dyn_dtree[d_code(dist - 1)].fc += 1;
        self.syms.len() == SYM_END
    }

    fn flush_block(&mut self, last: bool) {
        let stored = &self.input[self.block_start..self.strstart];
        self.trees.flush_block(&mut self.w, &self.syms, stored, last);
        self.syms.clear();
        self.block_start = self.strstart;
    }

    /// zlib `deflate_fast` (levels 1-3): greedy matching, short chains.
    fn deflate_fast(&mut self) {
        let max_insert = self.cfg.max_lazy;
        while self.strstart < self.input.len() {
            let lookahead = self.lookahead();
            let mut hash_head = NIL;
            if lookahead >= MIN_MATCH {
                hash_head = self.insert_string(self.strstart);
            }
            self.match_length = MIN_MATCH - 1;
            if hash_head != NIL && self.strstart - hash_head as usize <= MAX_DIST {
                self.prev_length = MIN_MATCH - 1;
                self.match_length = self.longest_match(hash_head as usize);
            }
            let bflush;
            if self.match_length >= MIN_MATCH {
                let ml = self.match_length;
                bflush = self.tally_dist(self.strstart - self.match_start, ml - MIN_MATCH);
                if ml <= max_insert && lookahead - ml >= MIN_MATCH {
                    // Insert every position of the match into the chain.
                    for _ in 1..ml {
                        self.strstart += 1;
                        self.insert_string(self.strstart);
                    }
                    self.strstart += 1;
                } else {
                    self.strstart += ml;
                }
                self.match_length = 0;
            } else {
                bflush = self.tally_lit(self.input[self.strstart]);
                self.strstart += 1;
            }
            if bflush {
                self.flush_block(false);
            }
        }
        self.flush_block(true);
    }

    /// zlib `deflate_slow` (levels 4-9): lazy evaluation.
    fn deflate_slow(&mut self) {
        let max_lazy = self.cfg.max_lazy;
        let n = self.input.len();
        while self.strstart < n {
            let lookahead = n - self.strstart;
            let mut hash_head = NIL;
            if lookahead >= MIN_MATCH {
                hash_head = self.insert_string(self.strstart);
            }
            self.prev_length = self.match_length;
            self.prev_match = self.match_start;
            self.match_length = MIN_MATCH - 1;
            if hash_head != NIL
                && self.prev_length < max_lazy
                && self.strstart - hash_head as usize <= MAX_DIST
            {
                self.match_length = self.longest_match(hash_head as usize);
                if self.match_length <= 5
                    && self.match_length == MIN_MATCH
                    && self.strstart - self.match_start > TOO_FAR
                {
                    self.match_length = MIN_MATCH - 1;
                }
            }
            if self.prev_length >= MIN_MATCH && self.match_length <= self.prev_length {
                let max_insert = self.strstart + lookahead - MIN_MATCH;
                let pl = self.prev_length;
                let bflush = self.tally_dist(self.strstart - 1 - self.prev_match, pl - MIN_MATCH);
                // Insert the match's remaining positions (strstart+1 ..).
                let mut cnt = pl - 2;
                while cnt != 0 {
                    self.strstart += 1;
                    if self.strstart <= max_insert {
                        self.insert_string(self.strstart);
                    }
                    cnt -= 1;
                }
                self.match_available = false;
                self.match_length = MIN_MATCH - 1;
                self.strstart += 1;
                if bflush {
                    self.flush_block(false);
                }
            } else if self.match_available {
                let bflush = self.tally_lit(self.input[self.strstart - 1]);
                if bflush {
                    self.flush_block_only();
                }
                self.strstart += 1;
            } else {
                self.match_available = true;
                self.strstart += 1;
            }
        }
        if self.match_available {
            self.tally_lit(self.input[self.strstart - 1]);
            self.match_available = false;
        }
        self.flush_block(true);
    }

    /// zlib `FLUSH_BLOCK_ONLY` inside the lazy literal path: the block ends
    /// at `strstart - 1` (that literal was the last symbol) — in zlib the
    /// stored range is `[block_start, strstart)` at that point too, since
    /// `strstart` has not been advanced yet.
    fn flush_block_only(&mut self) {
        self.flush_block(false);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn roundtrip(data: &[u8], level: u32) {
        let c = deflate(data, level);
        let mut out = Vec::new();
        super::super::inflate::inflate_into(&c, &mut out).unwrap();
        assert_eq!(out, data, "level {level} len {}", data.len());
        // Strict reference decoder.
        let mut dec = flate2::read::DeflateDecoder::new(&c[..]);
        let mut out2 = Vec::new();
        std::io::Read::read_to_end(&mut dec, &mut out2).unwrap();
        assert_eq!(out2, data, "miniz rejects level {level} len {}", data.len());
    }

    #[test]
    fn roundtrip_levels_and_shapes() {
        let text: Vec<u8> = b"the quick brown fox jumps over the lazy dog. ".repeat(3000);
        let mut rnd = Vec::new();
        let mut x = 0x1234_5678u32;
        for _ in 0..200_000 {
            x ^= x << 13;
            x ^= x >> 17;
            x ^= x << 5;
            rnd.push((x >> 24) as u8);
        }
        let zeros = vec![0u8; 300_000];
        let mut mixed = text.clone();
        mixed.extend_from_slice(&rnd[..50_000]);
        mixed.extend_from_slice(&zeros[..70_000]);
        for level in 0..=9 {
            for d in [&b""[..], b"a", b"abc", &text, &rnd, &zeros, &mixed] {
                roundtrip(d, level);
            }
        }
    }

    #[test]
    fn far_distances_are_within_window() {
        // A repeat exactly 32768 back is the largest legal distance.
        let mut d = vec![0u8; 0];
        let mut x = 7u32;
        for _ in 0..32768 {
            x = x.wrapping_mul(1103515245).wrapping_add(12345);
            d.push((x >> 16) as u8);
        }
        let first = d.clone();
        d.extend_from_slice(&first);
        for level in [1, 4, 6, 9] {
            roundtrip(&d, level);
        }
    }
}
