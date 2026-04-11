//! LZMA stream coder (decoder side first).
//!
//! Direct port of liblzma's `lzma_decoder.c` / `lzma_common.h`.  The model
//! has these main components:
//!
//! * **Literal sub-coder** — `2^(lc + lp)` × 0x300 probabilities.  Each
//!   literal is a context-mixed 8-bit symbol; if the previous LZMA event
//!   was a match, the literal is decoded against the "match byte" too.
//!
//! * **Length sub-coder** — encodes match lengths 2..=273 in three
//!   sub-trees of 8/8/256 symbols (low/mid/high), with two top-level
//!   choice bits.
//!
//! * **Distance sub-coder** — 64 distance "slots" indexed by the position
//!   state.  Slots 0..3 are direct, slots 4..13 use a per-slot reverse
//!   bittree, slots 14..63 use raw direct bits + a shared 16-entry
//!   alignment bittree.
//!
//! * **Repeat distances** — the four most recent match distances are
//!   remembered.  Decoder can choose to reuse one of them with a tiny
//!   probability cost instead of re-encoding the full distance.
//!
//! * **State machine** — 12 states tracking the last few events
//!   (literal / match / repeat / short repeat) which affect which
//!   probability subarrays are used.
//!
//! All numeric constants come straight from `lzma_common.h`.

use std::io;

use super::options::LzmaOptions;
use super::range_coder::{prob_init, prob_reset_slice, Prob, RangeDecoder, RangeEncoder};

// =========================================================================
// Constants (from lzma_common.h)
// =========================================================================

pub const LZMA_LCLP_MAX: u32 = 4;
pub const LZMA_PB_MAX: u32 = 4;

pub const POS_STATES_MAX: usize = 1 << LZMA_PB_MAX;

pub const STATES: usize = 12;
pub const LIT_STATES: u32 = 7;

pub const LITERAL_CODER_SIZE: usize = 0x300;
pub const LITERAL_CODERS_MAX: usize = 1 << LZMA_LCLP_MAX;

pub const MATCH_LEN_MIN: u32 = 2;

pub const LEN_LOW_BITS: u32 = 3;
pub const LEN_LOW_SYMBOLS: usize = 1 << LEN_LOW_BITS;
pub const LEN_MID_BITS: u32 = 3;
pub const LEN_MID_SYMBOLS: usize = 1 << LEN_MID_BITS;
pub const LEN_HIGH_BITS: u32 = 8;
pub const LEN_HIGH_SYMBOLS: usize = 1 << LEN_HIGH_BITS;
pub const LEN_SYMBOLS: usize = LEN_LOW_SYMBOLS + LEN_MID_SYMBOLS + LEN_HIGH_SYMBOLS;
pub const MATCH_LEN_MAX: u32 = MATCH_LEN_MIN + LEN_SYMBOLS as u32 - 1;

pub const DIST_STATES: u32 = 4;
pub const DIST_SLOT_BITS: u32 = 6;
pub const DIST_SLOTS: usize = 1 << DIST_SLOT_BITS;
pub const DIST_MODEL_START: u32 = 4;
pub const DIST_MODEL_END: u32 = 14;
pub const FULL_DISTANCES_BITS: u32 = DIST_MODEL_END / 2;
pub const FULL_DISTANCES: usize = 1 << FULL_DISTANCES_BITS;
pub const ALIGN_BITS: u32 = 4;
pub const ALIGN_SIZE: usize = 1 << ALIGN_BITS;

pub const REPS: usize = 4;

#[inline]
pub fn get_dist_state(len: u32) -> u32 {
    if len < DIST_STATES + MATCH_LEN_MIN {
        len - MATCH_LEN_MIN
    } else {
        DIST_STATES - 1
    }
}

#[inline]
pub fn is_literal_state(state: u32) -> bool {
    state < LIT_STATES
}

/// State transition: a literal was just emitted.
#[inline]
pub fn update_literal(state: u32) -> u32 {
    if state <= 3 {
        0
    } else if state <= 9 {
        state - 3
    } else {
        state - 6
    }
}

/// State transition: a non-repeat match was just emitted.
#[inline]
pub fn update_match(state: u32) -> u32 {
    if state < LIT_STATES {
        7
    } else {
        10
    }
}

/// State transition: a long repeat was just emitted.
#[inline]
pub fn update_long_rep(state: u32) -> u32 {
    if state < LIT_STATES {
        8
    } else {
        11
    }
}

/// State transition: a short repeat was just emitted.
#[inline]
pub fn update_short_rep(state: u32) -> u32 {
    if state < LIT_STATES {
        9
    } else {
        11
    }
}

// =========================================================================
// Length sub-coder
// =========================================================================

/// Length decoder: probabilities for the choice bits and three sub-trees.
pub struct LenDecoder {
    pub choice: Prob,
    pub choice2: Prob,
    pub low: [[Prob; LEN_LOW_SYMBOLS]; POS_STATES_MAX],
    pub mid: [[Prob; LEN_MID_SYMBOLS]; POS_STATES_MAX],
    pub high: [Prob; LEN_HIGH_SYMBOLS],
}

impl LenDecoder {
    pub fn new() -> Self {
        Self {
            choice: prob_init(),
            choice2: prob_init(),
            low: [[prob_init(); LEN_LOW_SYMBOLS]; POS_STATES_MAX],
            mid: [[prob_init(); LEN_MID_SYMBOLS]; POS_STATES_MAX],
            high: [prob_init(); LEN_HIGH_SYMBOLS],
        }
    }

    pub fn reset(&mut self) {
        self.choice = prob_init();
        self.choice2 = prob_init();
        for row in self.low.iter_mut() {
            prob_reset_slice(row);
        }
        for row in self.mid.iter_mut() {
            prob_reset_slice(row);
        }
        prob_reset_slice(&mut self.high);
    }

    /// Decode a match length, returning the *raw* length (not adjusted by
    /// MATCH_LEN_MIN).  Caller adds MATCH_LEN_MIN to get the actual length.
    #[inline]
    pub fn decode(&mut self, rd: &mut RangeDecoder, pos_state: usize) -> u32 {
        if rd.decode_bit_fast(&mut self.choice) == 0 {
            // 2..9
            rd.decode_bittree_fast(&mut self.low[pos_state], LEN_LOW_BITS)
        } else if rd.decode_bit_fast(&mut self.choice2) == 0 {
            // 10..17
            LEN_LOW_SYMBOLS as u32
                + rd.decode_bittree_fast(&mut self.mid[pos_state], LEN_MID_BITS)
        } else {
            // 18..273
            LEN_LOW_SYMBOLS as u32
                + LEN_MID_SYMBOLS as u32
                + rd.decode_bittree_fast(&mut self.high, LEN_HIGH_BITS)
        }
    }
}

// =========================================================================
// LZMA decoder
// =========================================================================

/// LZMA stream decoder.  Holds the entire probability model + the LZ
/// dictionary buffer.  Drives a `RangeDecoder` over a fixed input slice.
pub struct LzmaDecoder {
    // Coding parameters.
    pub lc: u32,
    pub lp: u32,
    pub pb: u32,
    pub lp_mask: u32,
    pub pb_mask: u32,

    // Probability arrays.
    /// `is_match[state * POS_STATES_MAX + pos_state]` — is the next event
    /// a literal (0) or some kind of match (1)?
    pub is_match: Vec<Prob>,
    pub is_rep: [Prob; STATES],
    pub is_rep0: [Prob; STATES],
    pub is_rep1: [Prob; STATES],
    pub is_rep2: [Prob; STATES],
    pub is_rep0_long: Vec<Prob>,

    /// Distance slot probabilities, `[dist_state][slot]`.
    pub dist_slot: [[Prob; DIST_SLOTS]; DIST_STATES as usize],
    /// Reverse-bittree probabilities for the middle-range distance slots.
    /// Sized as `FULL_DISTANCES` (not `FULL_DISTANCES - DIST_MODEL_END`)
    /// so the slot-13 slice `[base - slot..base - slot + 32]` fits cleanly.
    /// liblzma stores the array at the smaller size and walks via UB
    /// pointer arithmetic — we trade ~28 bytes for safe Rust slicing.
    pub dist_special: [Prob; FULL_DISTANCES],
    /// Alignment-bits reverse bittree for the very large distances.
    pub dist_align: [Prob; ALIGN_SIZE],

    pub match_len: LenDecoder,
    pub rep_len: LenDecoder,

    /// Literal probabilities, length `(1 << (lc + lp)) * LITERAL_CODER_SIZE`.
    pub literal: Vec<Prob>,

    // State machine.
    pub state: u32,
    pub reps: [u32; REPS],

    // Dictionary buffer (LZ77 sliding window).
    pub dict: Dict,
}

impl LzmaDecoder {
    pub fn new(lc: u32, lp: u32, pb: u32, dict_size: u32) -> io::Result<Self> {
        if lc > LZMA_LCLP_MAX || lp > LZMA_LCLP_MAX || lc + lp > LZMA_LCLP_MAX || pb > LZMA_PB_MAX {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!("lzma: invalid lc/lp/pb ({}, {}, {})", lc, lp, pb),
            ));
        }
        let literal_coders = 1usize << (lc + lp);
        Ok(Self {
            lc,
            lp,
            pb,
            lp_mask: (1u32 << lp) - 1,
            pb_mask: (1u32 << pb) - 1,
            is_match: vec![prob_init(); STATES * POS_STATES_MAX],
            is_rep: [prob_init(); STATES],
            is_rep0: [prob_init(); STATES],
            is_rep1: [prob_init(); STATES],
            is_rep2: [prob_init(); STATES],
            is_rep0_long: vec![prob_init(); STATES * POS_STATES_MAX],
            dist_slot: [[prob_init(); DIST_SLOTS]; DIST_STATES as usize],
            dist_special: [prob_init(); FULL_DISTANCES],
            dist_align: [prob_init(); ALIGN_SIZE],
            match_len: LenDecoder::new(),
            rep_len: LenDecoder::new(),
            literal: vec![prob_init(); literal_coders * LITERAL_CODER_SIZE],
            state: 0,
            reps: [0; REPS],
            dict: Dict::new(dict_size as usize),
            // shut up unused field reads:
            // (lp_mask/pb_mask are used in literal_subcoder + pos_state)
        })
    }

    /// Reset every probability and the state machine.  Does NOT clear the
    /// dictionary — LZMA2 controls dict reset separately via its chunk
    /// control byte.
    pub fn reset_state(&mut self) {
        prob_reset_slice(&mut self.is_match);
        prob_reset_slice(&mut self.is_rep);
        prob_reset_slice(&mut self.is_rep0);
        prob_reset_slice(&mut self.is_rep1);
        prob_reset_slice(&mut self.is_rep2);
        prob_reset_slice(&mut self.is_rep0_long);
        for row in self.dist_slot.iter_mut() {
            prob_reset_slice(row);
        }
        prob_reset_slice(&mut self.dist_special);
        prob_reset_slice(&mut self.dist_align);
        self.match_len.reset();
        self.rep_len.reset();
        prob_reset_slice(&mut self.literal);
        self.state = 0;
        self.reps = [0; REPS];
    }

    /// Compute the literal sub-coder index based on `lc`, `lp_mask`, the
    /// stream position (low bits) and the previous output byte.
    #[inline]
    fn literal_subcoder_offset(&self, pos_low: u32, prev_byte: u8) -> usize {
        // (((pos & lp_mask) << lc) + (prev_byte >> (8 - lc))) * LITERAL_CODER_SIZE
        let coder = ((pos_low & self.lp_mask) << self.lc)
            + ((prev_byte as u32) >> (8 - self.lc));
        (coder as usize) * LITERAL_CODER_SIZE
    }

    /// Decode one literal byte and write it to the dictionary.
    #[inline]
    fn decode_literal(&mut self, rd: &mut RangeDecoder) -> io::Result<()> {
        let pos_low = self.dict.position() as u32;
        let prev = self.dict.last_or_zero();
        let base = self.literal_subcoder_offset(pos_low, prev);

        let mut symbol: u32 = 1;
        // SAFETY: `base + symbol` and `base + ((1+match_bit) << 8) + symbol`
        // are always < `literal.len() = (1 << (lc + lp)) * LITERAL_CODER_SIZE`
        // because:
        //   * symbol is in [1, 0x1FF] during the loop (capped before each load)
        //   * base = coder * LITERAL_CODER_SIZE where coder < (1 << (lc + lp))
        //   * LITERAL_CODER_SIZE = 0x300, so each sub-coder slot owns
        //     indices [base, base + 0x300) which fully contains 0..0x300.
        let lit_ptr = self.literal.as_mut_ptr();

        if !is_literal_state(self.state) {
            // Match-byte mode: previous LZMA event was a match.  Use the
            // byte at distance reps[0]+1 from the current dict position
            // ("match byte") to influence which probability sub-tree we
            // descend into.
            let mut match_byte = self.dict.byte_at(self.reps[0] as usize + 1) as u32;
            loop {
                let match_bit = (match_byte >> 7) & 1;
                match_byte <<= 1;
                let prob_idx = base + ((1 + match_bit) << 8) as usize + symbol as usize;
                let bit = unsafe { rd.decode_bit_fast(&mut *lit_ptr.add(prob_idx)) };
                symbol = (symbol << 1) | bit;
                if match_bit != bit {
                    break;
                }
                if symbol >= 0x100 {
                    break;
                }
            }
        }

        // Continue (or do the entire) plain literal decode.
        while symbol < 0x100 {
            let prob_idx = base + symbol as usize;
            let bit = unsafe { rd.decode_bit_fast(&mut *lit_ptr.add(prob_idx)) };
            symbol = (symbol << 1) | bit;
        }

        self.dict.push(symbol as u8 & 0xff);
        self.state = update_literal(self.state);
        Ok(())
    }

    /// Decode the match-distance using the slot/direct/align scheme.
    #[inline]
    fn decode_distance(&mut self, rd: &mut RangeDecoder, len: u32) -> u32 {
        let dist_state = get_dist_state(len) as usize;
        let slot = rd.decode_bittree_fast(&mut self.dist_slot[dist_state], DIST_SLOT_BITS);
        if slot < DIST_MODEL_START {
            return slot;
        }
        let num_direct = (slot >> 1) - 1;
        let base: u32 = (2 | (slot & 1)) << num_direct;
        if slot < DIST_MODEL_END {
            let begin = base as usize - slot as usize;
            let end = begin + (1usize << num_direct);
            let probs = &mut self.dist_special[begin..end];
            let extra = rd.decode_bittree_reverse_fast(probs, num_direct);
            base + extra
        } else {
            let direct_bits = num_direct - ALIGN_BITS;
            let direct = rd.decode_direct_bits_fast(direct_bits) << ALIGN_BITS;
            let align = rd.decode_bittree_reverse_fast(&mut self.dist_align, ALIGN_BITS);
            base + direct + align
        }
    }

    /// Decode bytes from `input` until either:
    ///   (a) `uncompressed_remaining` bytes have been emitted, or
    ///   (b) the end-of-stream marker (distance == u32::MAX) is encountered.
    ///
    /// The caller MUST ensure `rd.input` has enough trailing bytes for the
    /// range decoder to refill freely.
    ///
    /// Returns `(bytes_emitted, hit_end_marker)`.
    pub fn decode_to_dict(
        &mut self,
        rd: &mut RangeDecoder,
        uncompressed_remaining: usize,
        output: &mut Vec<u8>,
    ) -> io::Result<(usize, bool)> {
        let mut produced = 0usize;
        let dict_cap = self.dict.buf.len();
        while produced < uncompressed_remaining {
            let pos_state = (self.dict.position() as u32 & self.pb_mask) as usize;
            let is_match_idx = self.state as usize * POS_STATES_MAX + pos_state;
            let bit = rd.decode_bit_fast(&mut self.is_match[is_match_idx]);

            if bit == 0 {
                // Literal.
                self.decode_literal(rd)?;
                // Mirror the freshly-written byte into the user output so
                // we never have to read it back out of the cyclic dict
                // (which may have wrapped on a long enough chunk).
                output.push(self.dict.byte_at(1));
                produced += 1;
                continue;
            }

            // Some kind of match.
            let len: u32;
            if rd.decode_bit_fast(&mut self.is_rep[self.state as usize]) != 0 {
                // Repeat.
                if rd.decode_bit_fast(&mut self.is_rep0[self.state as usize]) == 0 {
                    // rep0 — same distance as last time.
                    let rep0_long_idx =
                        self.state as usize * POS_STATES_MAX + pos_state;
                    if rd.decode_bit_fast(&mut self.is_rep0_long[rep0_long_idx]) == 0 {
                        // Short rep — exactly one byte.
                        self.state = update_short_rep(self.state);
                        let b = self.dict.byte_at(self.reps[0] as usize + 1);
                        self.dict.push(b);
                        output.push(b);
                        produced += 1;
                        continue;
                    }
                    len = MATCH_LEN_MIN + self.rep_len.decode(rd, pos_state);
                } else {
                    // rep1, rep2 or rep3.
                    let dist;
                    if rd.decode_bit_fast(&mut self.is_rep1[self.state as usize]) == 0 {
                        dist = self.reps[1];
                    } else {
                        if rd.decode_bit_fast(&mut self.is_rep2[self.state as usize]) == 0 {
                            dist = self.reps[2];
                        } else {
                            dist = self.reps[3];
                            self.reps[3] = self.reps[2];
                        }
                        self.reps[2] = self.reps[1];
                    }
                    self.reps[1] = self.reps[0];
                    self.reps[0] = dist;
                    len = MATCH_LEN_MIN + self.rep_len.decode(rd, pos_state);
                }
                self.state = update_long_rep(self.state);
            } else {
                // Plain match.  Shift the rep distances.
                self.reps[3] = self.reps[2];
                self.reps[2] = self.reps[1];
                self.reps[1] = self.reps[0];
                let raw_len = self.match_len.decode(rd, pos_state);
                len = MATCH_LEN_MIN + raw_len;
                self.reps[0] = self.decode_distance(rd, len);
                if self.reps[0] == u32::MAX {
                    // End-of-payload marker.
                    return Ok((produced, true));
                }
                self.state = update_match(self.state);
            }

            // Copy `len` bytes from the dictionary at offset `reps[0] + 1`.
            let copy_len = (len as usize).min(uncompressed_remaining - produced);
            self.dict.repeat(self.reps[0] as usize + 1, copy_len)?;
            // Mirror the freshly-written copy_len bytes into output.  Since
            // copy_len ≤ MATCH_LEN_MAX (273) ≤ dict_cap (xz min 4 KiB),
            // the just-written bytes are guaranteed to all still live in
            // the dict's cyclic buffer — at most a 2-slice wrap.
            let total_after = self.dict.total as usize;
            let start = (total_after - copy_len) % dict_cap;
            if start + copy_len <= dict_cap {
                output.extend_from_slice(&self.dict.buf[start..start + copy_len]);
            } else {
                let first = dict_cap - start;
                output.extend_from_slice(&self.dict.buf[start..dict_cap]);
                output.extend_from_slice(&self.dict.buf[..copy_len - first]);
            }
            produced += copy_len;
            if (copy_len as u32) < len {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidData,
                    "lzma: match copy overruns chunk size",
                ));
            }
        }
        Ok((produced, false))
    }
}

// =========================================================================
// LZ77 dictionary
// =========================================================================

/// Sliding-window LZ77 dictionary.  Holds the most recent `dict_size`
/// decoded bytes plus a write cursor.  We allocate the full configured
/// dict size up-front (xz dict_size is at most 4 GiB but in practice
/// preset 9 = 64 MiB).
pub struct Dict {
    pub buf: Vec<u8>,
    /// Total bytes ever written (used to compute the wrapped position).
    pub total: u64,
    /// `buf.len() - 1` when capacity is a power of two (fast modulo via
    /// bitwise AND), or 0 to signal that real modulo is needed.
    mask: usize,
}

impl Dict {
    pub fn new(size: usize) -> Self {
        let cap = size.max(1);
        let mut buf: Vec<u8> = Vec::with_capacity(cap);
        unsafe { buf.set_len(cap); }
        let mask = if cap.is_power_of_two() { cap - 1 } else { 0 };
        Self { buf, total: 0, mask }
    }

    pub fn capacity(&self) -> usize {
        self.buf.len()
    }

    /// Fast cyclic-buffer index: uses bitwise AND for power-of-2 buffers.
    #[inline(always)]
    fn wrap(&self, pos: usize) -> usize {
        if self.mask != 0 { pos & self.mask } else { pos % self.buf.len() }
    }

    /// Position within the cyclic buffer for the NEXT byte to be written.
    pub fn position(&self) -> usize {
        self.wrap(self.total as usize)
    }

    /// Most recently written byte (or 0 if dict is empty).
    pub fn last_or_zero(&self) -> u8 {
        if self.total == 0 {
            0
        } else {
            let pos = self.wrap((self.total - 1) as usize);
            self.buf[pos]
        }
    }

    /// Read the byte at `back_offset` bytes BEFORE the current write
    /// position.  `back_offset == 1` is the byte just written.  Returns 0
    /// if `back_offset` exceeds the bytes ever written.
    #[inline(always)]
    pub fn byte_at(&self, back_offset: usize) -> u8 {
        if back_offset == 0 || (back_offset as u64) > self.total {
            return 0;
        }
        let pos = self.wrap(self.total as usize + self.buf.len() - back_offset);
        unsafe { *self.buf.as_ptr().add(pos) }
    }

    #[inline(always)]
    pub fn push(&mut self, b: u8) {
        let pos = self.wrap(self.total as usize);
        unsafe { *self.buf.as_mut_ptr().add(pos) = b; }
        self.total += 1;
    }

    /// Repeat `len` bytes from `back_offset` bytes ago — the LZ77 copy.
    /// `back_offset` must be in `1..=total` and `1..=capacity`.
    pub fn repeat(&mut self, back_offset: usize, len: usize) -> io::Result<()> {
        if back_offset == 0 || (back_offset as u64) > self.total || back_offset > self.buf.len() {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!(
                    "lzma: distance {} out of range (total={}, cap={})",
                    back_offset,
                    self.total,
                    self.buf.len()
                ),
            ));
        }
        let cap = self.buf.len();
        let mut dst = self.wrap(self.total as usize);
        let mut src = self.wrap(self.total as usize + cap - back_offset);

        // Fast path: neither src nor dst wraps the cyclic buffer.
        let no_wrap_dst = dst + len <= cap;
        let no_wrap_src = src + len <= cap;
        if no_wrap_dst && no_wrap_src {
            unsafe {
                let p = self.buf.as_mut_ptr();
                if back_offset == 1 {
                    let b = *p.add(src);
                    std::ptr::write_bytes(p.add(dst), b, len);
                } else if back_offset >= len {
                    std::ptr::copy_nonoverlapping(p.add(src), p.add(dst), len);
                } else {
                    // Overlapping: copy in back_offset-sized chunks so
                    // the repeating pattern propagates correctly.
                    let mut w = 0usize;
                    while w + back_offset <= len {
                        std::ptr::copy_nonoverlapping(
                            p.add(src + w), p.add(dst + w), back_offset,
                        );
                        w += back_offset;
                    }
                    if w < len {
                        std::ptr::copy_nonoverlapping(
                            p.add(src + w), p.add(dst + w), len - w,
                        );
                    }
                }
            }
            self.total += len as u64;
            return Ok(());
        }

        // Slow path: at least one side wraps.  Uses wrap() (bitwise AND
        // for power-of-2 dicts) instead of modulo per byte.
        for _ in 0..len {
            let s = self.wrap(src);
            let d = self.wrap(dst);
            unsafe {
                let p = self.buf.as_mut_ptr();
                *p.add(d) = *p.add(s);
            }
            src += 1;
            dst += 1;
            self.total += 1;
        }
        Ok(())
    }
}

// =========================================================================
// LZMA encoder
// =========================================================================
//
// The encoder mirrors the decoder's probability model exactly — same
// arrays, same constants, same state machine — but uses `RangeEncoder`
// instead of `RangeDecoder` and *makes choices* about whether each byte
// is best emitted as a literal, a match, a "short rep" (1-byte repeat at
// the most-recent distance) or a "long rep" (longer match at one of the
// last 4 distances).
//
// MVP parser: greedy.  At each position, try to find the longest match in
// the dictionary via the hash-chain match finder; if it's at least 2 bytes
// long, use it.  Otherwise emit a literal.  This is correct (the decoder
// can recover the input bit-for-bit) but suboptimal for compression ratio
// — a near-optimal parser comes later.

/// Length encoder (mirror of `LenDecoder`).
pub struct LenEncoder {
    pub choice: Prob,
    pub choice2: Prob,
    pub low: [[Prob; LEN_LOW_SYMBOLS]; POS_STATES_MAX],
    pub mid: [[Prob; LEN_MID_SYMBOLS]; POS_STATES_MAX],
    pub high: [Prob; LEN_HIGH_SYMBOLS],
}

impl LenEncoder {
    pub fn new() -> Self {
        Self {
            choice: prob_init(),
            choice2: prob_init(),
            low: [[prob_init(); LEN_LOW_SYMBOLS]; POS_STATES_MAX],
            mid: [[prob_init(); LEN_MID_SYMBOLS]; POS_STATES_MAX],
            high: [prob_init(); LEN_HIGH_SYMBOLS],
        }
    }

    /// Encode a length, where `raw_len = actual_len - MATCH_LEN_MIN` (so the
    /// raw symbol fits in 0..=LEN_SYMBOLS-1).
    pub fn encode(&mut self, rc: &mut RangeEncoder, pos_state: usize, raw_len: u32) {
        if raw_len < LEN_LOW_SYMBOLS as u32 {
            rc.encode_bit(&mut self.choice, 0);
            rc.encode_bittree(&mut self.low[pos_state], LEN_LOW_BITS, raw_len);
        } else if raw_len < (LEN_LOW_SYMBOLS + LEN_MID_SYMBOLS) as u32 {
            rc.encode_bit(&mut self.choice, 1);
            rc.encode_bit(&mut self.choice2, 0);
            rc.encode_bittree(
                &mut self.mid[pos_state],
                LEN_MID_BITS,
                raw_len - LEN_LOW_SYMBOLS as u32,
            );
        } else {
            rc.encode_bit(&mut self.choice, 1);
            rc.encode_bit(&mut self.choice2, 1);
            rc.encode_bittree(
                &mut self.high,
                LEN_HIGH_BITS,
                raw_len - (LEN_LOW_SYMBOLS + LEN_MID_SYMBOLS) as u32,
            );
        }
    }
}

/// LZMA encoder.  Operates over a fully-buffered input slice and writes
/// the range-coded output into its internal `RangeEncoder`.  Caller is
/// responsible for the LZMA2 chunk framing on top.
pub struct LzmaEncoder {
    // Coding parameters.
    pub lc: u32,
    pub lp: u32,
    pub pb: u32,
    lp_mask: u32,
    pb_mask: u32,
    nice_len: u32,
    dict_size: u32,

    // Probability arrays — exactly the same shape as `LzmaDecoder`.
    is_match: Vec<Prob>,
    is_rep: [Prob; STATES],
    is_rep0: [Prob; STATES],
    is_rep1: [Prob; STATES],
    is_rep2: [Prob; STATES],
    is_rep0_long: Vec<Prob>,
    dist_slot: [[Prob; DIST_SLOTS]; DIST_STATES as usize],
    dist_special: [Prob; FULL_DISTANCES],
    dist_align: [Prob; ALIGN_SIZE],
    match_len: LenEncoder,
    rep_len: LenEncoder,
    literal: Vec<Prob>,

    // State machine.
    state: u32,
    reps: [u32; REPS],

    // Range encoder buffer (one per LZMA2 chunk).
    pub rc: RangeEncoder,
}



impl LzmaEncoder {
    pub fn new(lc: u32, lp: u32, pb: u32, dict_size: u32, nice_len: u32) -> io::Result<Self> {
        if lc > LZMA_LCLP_MAX || lp > LZMA_LCLP_MAX || lc + lp > LZMA_LCLP_MAX || pb > LZMA_PB_MAX
        {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!("lzma: invalid lc/lp/pb ({}, {}, {})", lc, lp, pb),
            ));
        }
        let literal_coders = 1usize << (lc + lp);
        Ok(Self {
            lc,
            lp,
            pb,
            lp_mask: (1u32 << lp) - 1,
            pb_mask: (1u32 << pb) - 1,
            nice_len,
            dict_size,
            is_match: vec![prob_init(); STATES * POS_STATES_MAX],
            is_rep: [prob_init(); STATES],
            is_rep0: [prob_init(); STATES],
            is_rep1: [prob_init(); STATES],
            is_rep2: [prob_init(); STATES],
            is_rep0_long: vec![prob_init(); STATES * POS_STATES_MAX],
            dist_slot: [[prob_init(); DIST_SLOTS]; DIST_STATES as usize],
            dist_special: [prob_init(); FULL_DISTANCES],
            dist_align: [prob_init(); ALIGN_SIZE],
            match_len: LenEncoder::new(),
            rep_len: LenEncoder::new(),
            literal: vec![prob_init(); literal_coders * LITERAL_CODER_SIZE],
            state: 0,
            reps: [0; REPS],
            rc: RangeEncoder::new(),
        })
    }

    /// Reset just the range coder buffer (between LZMA2 chunks).
    pub fn reset_range_coder(&mut self) {
        self.rc.reset();
    }

    /// Encode `len` input bytes starting at `input[start..]`, where
    /// `start` is the absolute position in the input stream (used both as
    /// the literal sub-coder context input and to compute `pos_state`).
    ///
    /// Hand-rolled greedy parser: at each position scan the hash chain
    /// for the longest match.  If `len >= MATCH_LEN_MIN`, emit a match;
    /// otherwise emit a literal.
    pub fn encode_chunk(
        &mut self,
        input: &[u8],
        start: usize,
        len: usize,
        mf: &mut HashChain,
    ) -> io::Result<()> {
        let end = start + len;
        let mut pos = start;

        while pos < end {
            let pos_state = (pos as u32 & self.pb_mask) as usize;
            let is_match_idx = self.state as usize * POS_STATES_MAX + pos_state;

            // Look for matches via the hash chain.
            let (best_match_len, best_match_dist) = if pos + MATCH_LEN_MIN as usize <= end {
                mf.find_longest(input, pos, end, self.dict_size as usize, self.nice_len)
            } else {
                (0, 0)
            };

            // Look for repeat-distance matches.
            let mut best_rep_idx: i32 = -1;
            let mut best_rep_len: u32 = 0;
            for i in 0..REPS {
                let d = self.reps[i] + 1;
                let rep_len = match_length(input, pos, end, d as usize);
                if rep_len > best_rep_len && rep_len >= MATCH_LEN_MIN {
                    best_rep_len = rep_len;
                    best_rep_idx = i as i32;
                }
            }
            // Even a 1-byte rep0 (the "short rep") can be cheaper than a
            // literal if reps[0]+1 byte matches the current.
            let mut short_rep_possible = false;
            if best_rep_idx < 0 {
                let d = self.reps[0] + 1;
                if (d as usize) <= pos && input[pos - d as usize] == input[pos] {
                    short_rep_possible = true;
                }
            }

            // Pick: rep > match if rep length is within 1 of match length,
            // since rep encoding is cheaper.  Otherwise use the longest.
            let use_rep = best_rep_idx >= 0
                && (best_match_len < MATCH_LEN_MIN || best_rep_len + 1 >= best_match_len);
            let use_match = !use_rep && best_match_len >= MATCH_LEN_MIN;
            let use_short_rep = !use_rep && !use_match && short_rep_possible;

            if use_rep {
                // Long rep match.
                self.rc.encode_bit(&mut self.is_match[is_match_idx], 1);
                self.rc.encode_bit(&mut self.is_rep[self.state as usize], 1);
                let rep_idx = best_rep_idx as usize;
                if rep_idx == 0 {
                    self.rc.encode_bit(&mut self.is_rep0[self.state as usize], 0);
                    let rep0_long_idx =
                        self.state as usize * POS_STATES_MAX + pos_state;
                    self.rc.encode_bit(&mut self.is_rep0_long[rep0_long_idx], 1);
                } else {
                    self.rc.encode_bit(&mut self.is_rep0[self.state as usize], 1);
                    if rep_idx == 1 {
                        self.rc.encode_bit(&mut self.is_rep1[self.state as usize], 0);
                    } else {
                        self.rc.encode_bit(&mut self.is_rep1[self.state as usize], 1);
                        if rep_idx == 2 {
                            self.rc.encode_bit(&mut self.is_rep2[self.state as usize], 0);
                        } else {
                            self.rc.encode_bit(&mut self.is_rep2[self.state as usize], 1);
                        }
                    }
                    // Shuffle reps so that reps[0] = the chosen distance.
                    let chosen = self.reps[rep_idx];
                    for j in (1..=rep_idx).rev() {
                        self.reps[j] = self.reps[j - 1];
                    }
                    self.reps[0] = chosen;
                }
                let raw_len = best_rep_len - MATCH_LEN_MIN;
                self.rep_len.encode(&mut self.rc, pos_state, raw_len);
                self.state = update_long_rep(self.state);
                for k in 0..best_rep_len as usize {
                    if pos + k < end {
                        mf.insert(input, pos + k);
                    }
                }
                pos += best_rep_len as usize;
            } else if use_match {
                self.rc.encode_bit(&mut self.is_match[is_match_idx], 1);
                self.rc.encode_bit(&mut self.is_rep[self.state as usize], 0);
                let raw_len = best_match_len - MATCH_LEN_MIN;
                self.match_len.encode(&mut self.rc, pos_state, raw_len);
                let dist_value = best_match_dist - 1;
                self.encode_distance(dist_value, best_match_len);
                self.reps[3] = self.reps[2];
                self.reps[2] = self.reps[1];
                self.reps[1] = self.reps[0];
                self.reps[0] = dist_value;
                self.state = update_match(self.state);
                for k in 0..best_match_len as usize {
                    if pos + k < end {
                        mf.insert(input, pos + k);
                    }
                }
                pos += best_match_len as usize;
            } else if use_short_rep {
                self.rc.encode_bit(&mut self.is_match[is_match_idx], 1);
                self.rc.encode_bit(&mut self.is_rep[self.state as usize], 1);
                self.rc.encode_bit(&mut self.is_rep0[self.state as usize], 0);
                let rep0_long_idx =
                    self.state as usize * POS_STATES_MAX + pos_state;
                self.rc.encode_bit(&mut self.is_rep0_long[rep0_long_idx], 0);
                self.state = update_short_rep(self.state);
                mf.insert(input, pos);
                pos += 1;
            } else {
                // Literal.
                self.rc.encode_bit(&mut self.is_match[is_match_idx], 0);
                self.encode_literal(input, pos);
                self.state = update_literal(self.state);
                mf.insert(input, pos);
                pos += 1;
            }
        }
        Ok(())
    }

    /// Encode a literal byte at absolute position `pos` against the
    /// appropriate sub-coder.  When the previous LZMA event was a match
    /// we use the "match byte" mode that conditions each bit on the
    /// equivalent bit of the byte at distance reps[0]+1.
    fn encode_literal(&mut self, input: &[u8], pos: usize) {
        let cur = input[pos];
        let prev = if pos == 0 { 0u8 } else { input[pos - 1] };
        let coder = (((pos as u32) & self.lp_mask) << self.lc)
            + ((prev as u32) >> (8 - self.lc));
        let base = (coder as usize) * LITERAL_CODER_SIZE;

        let mut symbol: u32 = 1;
        if !is_literal_state(self.state) {
            // Match byte = the byte we'd be matching against if we had
            // emitted a rep0.  i.e. input[pos - reps[0] - 1].
            let d = (self.reps[0] + 1) as usize;
            let match_byte = if d <= pos { input[pos - d] } else { 0 };
            let mut mb = match_byte as u32;
            loop {
                let match_bit = (mb >> 7) & 1;
                mb <<= 1;
                let prob_idx = base + ((1 + match_bit) << 8) as usize + symbol as usize;
                let bit = ((cur as u32) >> (7 - bit_index(symbol))) & 1;
                self.rc.encode_bit(&mut self.literal[prob_idx], bit);
                symbol = (symbol << 1) | bit;
                if match_bit != bit {
                    break;
                }
                if symbol >= 0x100 {
                    return;
                }
            }
        }
        while symbol < 0x100 {
            let prob_idx = base + symbol as usize;
            let bit = ((cur as u32) >> (7 - bit_index(symbol))) & 1;
            self.rc.encode_bit(&mut self.literal[prob_idx], bit);
            symbol = (symbol << 1) | bit;
        }
    }

    /// Encode the match distance for a match of length `len`.  The encoded
    /// `dist_value` is `actual_distance - 1`.
    fn encode_distance(&mut self, dist_value: u32, len: u32) {
        let dist_state = get_dist_state(len) as usize;
        let slot = compute_dist_slot(dist_value);
        self.rc
            .encode_bittree(&mut self.dist_slot[dist_state], DIST_SLOT_BITS, slot);

        if slot >= DIST_MODEL_START {
            let num_direct = (slot >> 1) - 1;
            let base: u32 = (2 | (slot & 1)) << num_direct;
            let extra = dist_value - base;
            if slot < DIST_MODEL_END {
                let begin = base as usize - slot as usize;
                let end = begin + (1usize << num_direct);
                let probs = &mut self.dist_special[begin..end];
                self.rc
                    .encode_bittree_reverse(probs, num_direct, extra);
            } else {
                let direct_bits = num_direct - ALIGN_BITS;
                let direct = extra >> ALIGN_BITS;
                let align = extra & ((1 << ALIGN_BITS) - 1);
                self.rc.encode_direct_bits(direct, direct_bits);
                self.rc
                    .encode_bittree_reverse(&mut self.dist_align, ALIGN_BITS, align);
            }
        }
    }

}

/// Compute the LZMA distance slot from a 0-based distance value.
/// Slot is `floor(log2(dist_value)) * 2 + bit_below_top`, capped at 63.
fn compute_dist_slot(dist: u32) -> u32 {
    if dist < DIST_MODEL_START {
        return dist;
    }
    let bits = 31 - dist.leading_zeros();
    // top two bits are always 1x; bit (bits-1) is the low bit of the slot.
    (bits * 2) + ((dist >> (bits - 1)) & 1)
}

/// Bit index 0..7 for the next bit to encode in a literal.  When `symbol`
/// is in [1, 0xFF], the next bit to encode is at position
/// `(7 - log2(symbol))`.  This relies on `symbol` being shaped 1, 1x, 1xx, ...
#[inline]
fn bit_index(symbol: u32) -> u32 {
    // log2 of symbol — for the canonical "build the byte left to right"
    // walk, where symbol grows as 1, 2|3, 4|5|6|7, ...
    31 - symbol.leading_zeros()
}

/// Match length: how many consecutive bytes are equal between `input[pos]`
/// and `input[pos - dist]`.  Capped at `MATCH_LEN_MAX`.
pub fn match_length(input: &[u8], pos: usize, end: usize, dist: usize) -> u32 {
    if dist == 0 || dist > pos {
        return 0;
    }
    let max = (end - pos).min(MATCH_LEN_MAX as usize);
    let src = pos - dist;
    let mut k = 0usize;
    while k < max && input[pos + k] == input[src + k] {
        k += 1;
    }
    k as u32
}

// =========================================================================
// Hash-chain match finder
// =========================================================================
//
// Simple HC4-style match finder.  Hash function: 4-byte rolling hash.
// `head[h]` = most recent position whose 4-byte prefix hashes to `h`.
// `chain[i]` = previous position in the same hash bucket.

const HASH_BITS: u32 = 16;
const HASH_SIZE: usize = 1 << HASH_BITS;
const HASH_MASK: u32 = (HASH_SIZE as u32) - 1;
const NIL: u32 = u32::MAX;

pub struct HashChain {
    head: Vec<u32>,
    chain: Vec<u32>,
    /// Same `dict_size` as the LZMA encoder; entries older than
    /// `pos - dict_size` are out of bounds and skipped.
    pub dict_size: usize,
    /// Maximum chain length to walk before giving up — bounds the
    /// per-position cost.
    pub max_chain: u32,
}

impl HashChain {
    pub fn new(input_len: usize, dict_size: usize, max_chain: u32) -> Self {
        Self {
            head: vec![NIL; HASH_SIZE],
            chain: vec![NIL; input_len.max(1)],
            dict_size,
            max_chain,
        }
    }

    #[inline]
    fn hash4(b: &[u8]) -> u32 {
        let v = (b[0] as u32)
            | ((b[1] as u32) << 8)
            | ((b[2] as u32) << 16)
            | ((b[3] as u32) << 24);
        // FNV-ish mix.
        let h = v.wrapping_mul(0x9E3779B1);
        (h >> (32 - HASH_BITS)) & HASH_MASK
    }

    pub fn insert(&mut self, input: &[u8], pos: usize) {
        if pos + 4 > input.len() {
            return;
        }
        let h = Self::hash4(&input[pos..pos + 4]) as usize;
        self.chain[pos] = self.head[h];
        self.head[h] = pos as u32;
    }

    /// Walk the hash chain at position `pos` and return the longest match
    /// found.  `nice_len` is an early-out: stop searching once any match
    /// reaches this length.
    pub fn find_longest(
        &mut self,
        input: &[u8],
        pos: usize,
        end: usize,
        dict_size: usize,
        nice_len: u32,
    ) -> (u32, u32) {
        if pos + 4 > end {
            return (0, 0);
        }
        let h = Self::hash4(&input[pos..pos + 4]) as usize;
        let mut chain_pos = self.head[h];
        let earliest = pos.saturating_sub(dict_size);
        let mut best_len: u32 = 0;
        let mut best_dist: u32 = 0;
        let mut tries = self.max_chain;
        while chain_pos != NIL && (chain_pos as usize) >= earliest && tries > 0 {
            tries -= 1;
            let cand_pos = chain_pos as usize;
            // Quick reject: 4 bytes must match.
            if input[cand_pos] == input[pos]
                && input[cand_pos + 1] == input[pos + 1]
                && input[cand_pos + 2] == input[pos + 2]
                && input[cand_pos + 3] == input[pos + 3]
            {
                let dist = (pos - cand_pos) as u32;
                let len = match_length(input, pos, end, dist as usize);
                if len > best_len {
                    best_len = len;
                    best_dist = dist;
                    if len >= nice_len {
                        break;
                    }
                }
            }
            chain_pos = self.chain[cand_pos];
        }
        (best_len, best_dist)
    }
}

// =========================================================================
// Top-level: LZMA → LZMA2 wrapper
// =========================================================================

/// Encode `input` as an LZMA2 stream (the inner payload of an .xz block).
/// Single LZMA2 chunk per ~16 KiB of uncompressed input — chosen so that
/// the worst-case compressed size always fits in the 64 KiB chunk limit.
pub fn encode_lzma_to_lzma2(
    input: &[u8],
    options: &LzmaOptions,
    output: &mut Vec<u8>,
) -> io::Result<()> {
    if input.is_empty() {
        // An empty stream is just an end marker.
        output.push(0x00);
        return Ok(());
    }

    let lc = options.lc;
    let lp = options.lp;
    let pb = options.pb;
    let nice_len = options.nice_len.max(MATCH_LEN_MIN).min(MATCH_LEN_MAX);
    // Cap dict_size at the input length so the match finder never wastes
    // chain walks on entries that fall outside the dict window.
    let dict_size = options.dict_size.min(u32::MAX) as u32;

    let mut encoder = LzmaEncoder::new(lc, lp, pb, dict_size, nice_len)?;
    let mut mf = HashChain::new(input.len(), dict_size as usize, /* max_chain */ 32);

    // Chunk by uncompressed size.  16 KiB keeps the worst-case LZMA chunk
    // compressed size well under the 64 KiB LZMA2 limit.
    const CHUNK: usize = 16 * 1024;
    let mut pos = 0usize;
    let mut first = true;
    while pos < input.len() {
        let take = (input.len() - pos).min(CHUNK);

        encoder.reset_range_coder();
        encoder.encode_chunk(input, pos, take, &mut mf)?;
        encoder.rc.finish();
        let compressed = std::mem::take(&mut encoder.rc.output);

        if compressed.len() > 0xFFFF + 1 {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!(
                    "lzma2: chunk compressed size {} exceeds 64 KiB limit",
                    compressed.len()
                ),
            ));
        }

        // ---- Write LZMA2 chunk header ----
        // First chunk uses mode 3 (new props + dict reset = 0xE0).
        // Later chunks use mode 0 (no reset = 0x80) so the probability
        // model carries forward.
        let unc_minus_1 = (take - 1) as u32;
        let comp_minus_1 = (compressed.len() - 1) as u32;
        let unc_high_5 = ((unc_minus_1 >> 16) & 0x1F) as u8;
        let control = if first { 0xE0 } else { 0x80 } | unc_high_5;
        output.push(control);
        output.push(((unc_minus_1 >> 8) & 0xFF) as u8);
        output.push((unc_minus_1 & 0xFF) as u8);
        output.push(((comp_minus_1 >> 8) & 0xFF) as u8);
        output.push((comp_minus_1 & 0xFF) as u8);
        if first {
            // Properties byte: (pb * 5 + lp) * 9 + lc
            let props = (pb * 5 + lp) * 9 + lc;
            output.push(props as u8);
        }
        output.extend_from_slice(&compressed);

        pos += take;
        first = false;
    }

    // End-of-stream marker.
    output.push(0x00);
    Ok(())
}
