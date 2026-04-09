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
use super::range_coder::{prob_init, prob_reset_slice, Prob, RangeDecoder};

/// MVP encoder stub — wired in by `xz_format::encode_xz_stream` so the
/// container code compiles before the actual LZMA encoder lands.  Returns
/// an `Unsupported` error until the real encoder is implemented.
pub fn encode_lzma_to_lzma2(
    _input: &[u8],
    _options: &LzmaOptions,
    _output: &mut Vec<u8>,
) -> io::Result<()> {
    Err(io::Error::new(
        io::ErrorKind::Unsupported,
        "lzma: pure-Rust encoder not yet implemented",
    ))
}

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
    pub fn decode(&mut self, rd: &mut RangeDecoder, pos_state: usize) -> io::Result<u32> {
        if rd.decode_bit(&mut self.choice)? == 0 {
            // 2..9
            rd.decode_bittree(&mut self.low[pos_state], LEN_LOW_BITS)
        } else if rd.decode_bit(&mut self.choice2)? == 0 {
            // 10..17
            Ok(LEN_LOW_SYMBOLS as u32
                + rd.decode_bittree(&mut self.mid[pos_state], LEN_MID_BITS)?)
        } else {
            // 18..273
            Ok(LEN_LOW_SYMBOLS as u32
                + LEN_MID_SYMBOLS as u32
                + rd.decode_bittree(&mut self.high, LEN_HIGH_BITS)?)
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
    fn decode_literal(&mut self, rd: &mut RangeDecoder) -> io::Result<()> {
        let pos_low = self.dict.position() as u32;
        let prev = self.dict.last_or_zero();
        let base = self.literal_subcoder_offset(pos_low, prev);

        let mut symbol: u32 = 1;

        if !is_literal_state(self.state) {
            // Match-byte mode: previous LZMA event was a match.  Use the
            // byte at distance reps[0]+1 from the current dict position
            // ("match byte") to influence which probability sub-tree we
            // descend into.
            let match_byte = self.dict.byte_at(self.reps[0] as usize + 1);
            let mut match_byte = match_byte as u32;
            loop {
                let match_bit = (match_byte >> 7) & 1;
                match_byte <<= 1;
                let prob_idx = base + ((1 + match_bit) << 8) as usize + symbol as usize;
                let bit = rd.decode_bit(&mut self.literal[prob_idx])?;
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
            let bit = rd.decode_bit(&mut self.literal[prob_idx])?;
            symbol = (symbol << 1) | bit;
        }

        self.dict.push(symbol as u8 & 0xff);
        self.state = update_literal(self.state);
        Ok(())
    }

    /// Decode the match-distance using the slot/direct/align scheme.
    fn decode_distance(&mut self, rd: &mut RangeDecoder, len: u32) -> io::Result<u32> {
        let dist_state = get_dist_state(len) as usize;
        let slot = rd.decode_bittree(&mut self.dist_slot[dist_state], DIST_SLOT_BITS)?;
        if slot < DIST_MODEL_START {
            return Ok(slot);
        }
        let num_direct = (slot >> 1) - 1;
        let base: u32 = (2 | (slot & 1)) << num_direct;
        if slot < DIST_MODEL_END {
            // Reverse bittree decode against the slice of `dist_special`
            // beginning at `base - slot`.
            let begin = base as usize - slot as usize;
            let end = begin + (1usize << num_direct);
            let probs = &mut self.dist_special[begin..end];
            let extra = rd_decode_reverse(rd, probs, num_direct)?;
            Ok(base + extra)
        } else {
            // Direct bits + alignment bits.
            let direct_bits = num_direct - ALIGN_BITS;
            let direct = rd.decode_direct_bits(direct_bits)? << ALIGN_BITS;
            let align = rd.decode_bittree_reverse(&mut self.dist_align, ALIGN_BITS)?;
            Ok(base + direct + align)
        }
    }

    /// Decode bytes from `input` into the dictionary until either:
    /// (a) `uncompressed_remaining` bytes have been emitted, or
    /// (b) the end-of-stream marker (distance == u32::MAX) is encountered, or
    /// (c) the input is exhausted (returns Ok with however many bytes were emitted).
    ///
    /// Returns `(bytes_emitted, hit_end_marker)`.
    pub fn decode_to_dict(
        &mut self,
        rd: &mut RangeDecoder,
        uncompressed_remaining: usize,
    ) -> io::Result<(usize, bool)> {
        let mut produced = 0usize;
        while produced < uncompressed_remaining {
            let pos_state = (self.dict.position() as u32 & self.pb_mask) as usize;
            let is_match_idx = self.state as usize * POS_STATES_MAX + pos_state;
            let bit = rd.decode_bit(&mut self.is_match[is_match_idx])?;

            if bit == 0 {
                // Literal.
                self.decode_literal(rd)?;
                produced += 1;
                continue;
            }

            // Some kind of match.
            let len: u32;
            if rd.decode_bit(&mut self.is_rep[self.state as usize])? != 0 {
                // Repeat.
                if rd.decode_bit(&mut self.is_rep0[self.state as usize])? == 0 {
                    // rep0 — same distance as last time.
                    let rep0_long_idx =
                        self.state as usize * POS_STATES_MAX + pos_state;
                    if rd.decode_bit(&mut self.is_rep0_long[rep0_long_idx])? == 0 {
                        // Short rep — exactly one byte.
                        self.state = update_short_rep(self.state);
                        let b = self.dict.byte_at(self.reps[0] as usize + 1);
                        self.dict.push(b);
                        produced += 1;
                        continue;
                    }
                    len = MATCH_LEN_MIN + self.rep_len.decode(rd, pos_state)?;
                } else {
                    // rep1, rep2 or rep3.
                    let dist;
                    if rd.decode_bit(&mut self.is_rep1[self.state as usize])? == 0 {
                        dist = self.reps[1];
                    } else {
                        if rd.decode_bit(&mut self.is_rep2[self.state as usize])? == 0 {
                            dist = self.reps[2];
                        } else {
                            dist = self.reps[3];
                            self.reps[3] = self.reps[2];
                        }
                        self.reps[2] = self.reps[1];
                    }
                    self.reps[1] = self.reps[0];
                    self.reps[0] = dist;
                    len = MATCH_LEN_MIN + self.rep_len.decode(rd, pos_state)?;
                }
                self.state = update_long_rep(self.state);
            } else {
                // Plain match.  Shift the rep distances.
                self.reps[3] = self.reps[2];
                self.reps[2] = self.reps[1];
                self.reps[1] = self.reps[0];
                let raw_len = self.match_len.decode(rd, pos_state)?;
                len = MATCH_LEN_MIN + raw_len;
                // `get_dist_state` expects the full length (including the
                // MATCH_LEN_MIN offset), not the raw symbol from the length
                // decoder.
                self.reps[0] = self.decode_distance(rd, len)?;
                if self.reps[0] == u32::MAX {
                    // End-of-payload marker.
                    return Ok((produced, true));
                }
                self.state = update_match(self.state);
            }

            // Copy `len` bytes from the dictionary at offset `reps[0] + 1`.
            // Cap at the remaining requested-output budget so we never
            // overrun an LZMA2 chunk.
            let copy_len = (len as usize).min(uncompressed_remaining - produced);
            self.dict.repeat(self.reps[0] as usize + 1, copy_len)?;
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

/// Helper for the reverse-bittree decode against a `&mut [Prob]` slice.
/// Inlined to avoid borrow conflicts inside `LzmaDecoder::decode_distance`.
#[inline]
fn rd_decode_reverse(
    rd: &mut RangeDecoder,
    probs: &mut [Prob],
    num_bits: u32,
) -> io::Result<u32> {
    rd.decode_bittree_reverse(probs, num_bits)
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
}

impl Dict {
    pub fn new(size: usize) -> Self {
        // Always allocate at least 1 so byte_at() never indexes into an
        // empty Vec on a freshly-created decoder.
        let cap = size.max(1);
        Self {
            buf: vec![0u8; cap],
            total: 0,
        }
    }

    pub fn capacity(&self) -> usize {
        self.buf.len()
    }

    /// Position within the cyclic buffer for the NEXT byte to be written.
    pub fn position(&self) -> usize {
        (self.total as usize) % self.buf.len()
    }

    /// Most recently written byte (or 0 if dict is empty).
    pub fn last_or_zero(&self) -> u8 {
        if self.total == 0 {
            0
        } else {
            let pos = ((self.total - 1) as usize) % self.buf.len();
            self.buf[pos]
        }
    }

    /// Read the byte at `back_offset` bytes BEFORE the current write
    /// position.  `back_offset == 1` is the byte just written.  Returns 0
    /// if `back_offset` exceeds the bytes ever written.
    pub fn byte_at(&self, back_offset: usize) -> u8 {
        if back_offset == 0 || (back_offset as u64) > self.total {
            return 0;
        }
        let cap = self.buf.len();
        let pos = (self.total as usize + cap - back_offset) % cap;
        self.buf[pos]
    }

    pub fn push(&mut self, b: u8) {
        let cap = self.buf.len();
        let pos = (self.total as usize) % cap;
        self.buf[pos] = b;
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
        for _ in 0..len {
            let b = self.byte_at(back_offset);
            self.push(b);
        }
        Ok(())
    }
}
