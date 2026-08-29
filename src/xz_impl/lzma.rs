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
// Range decoder hot-path state
// =========================================================================

/// Range-decoder registers held in locals for the duration of one
/// `decode_into` call. `RangeDecoder`'s fields are copied in at entry and
/// written back at exit so the per-bit path never touches memory except
/// for the probability itself and the input byte on normalize.
///
/// Every method is `unsafe`: the caller guarantees `inp` has at least
/// `RC_PADDING`-style slack past the real data (see lzma2.rs) so the
/// normalize refill never needs a bounds check.
#[derive(Clone, Copy)]
struct Rc {
    range: u32,
    code: u32,
    pos: usize,
}

const RC_TOP: u32 = 1 << 24;
const RC_MODEL_BITS: u32 = 11;
const RC_MOVE: u32 = 5;
const RC_MODEL_OFFSET: u32 = (1 << RC_MOVE) - 1; // 31

/// All ones if `a < b`, else 0.
///
/// On x86-64 this is a two-instruction `cmp; sbb` asm block. The asm is
/// deliberately opaque: written in plain Rust, LLVM recognises the mask
/// pattern as a select, and its x86 cmov-conversion pass then turns the
/// selects back into *branches* inside the decode loop — cachegrind showed
/// ~45% of all decoder mispredicts on the "branchless" direct-bits path.
#[inline(always)]
fn lt_mask(a: u32, b: u32) -> u32 {
    #[cfg(target_arch = "x86_64")]
    {
        let mask: u32;
        // SAFETY: pure register arithmetic; `sbb m, m` yields -CF whatever
        // `m` held before, so the uninitialised output read is harmless.
        unsafe {
            core::arch::asm!(
                "cmp {a:e}, {b:e}",
                "sbb {m:e}, {m:e}",
                a = in(reg) a,
                b = in(reg) b,
                m = out(reg) mask,
                options(pure, nomem, nostack),
            );
        }
        mask
    }
    #[cfg(not(target_arch = "x86_64"))]
    {
        0u32.wrapping_sub((a < b) as u32)
    }
}

impl Rc {
    #[inline(always)]
    unsafe fn normalize(&mut self, inp: *const u8) {
        if self.range < RC_TOP {
            self.range <<= 8;
            self.code = (self.code << 8) | unsafe { *inp.add(self.pos) } as u32;
            self.pos += 1;
        }
    }

    /// Branchless bit decode (xz 5.6 `rc_c_bit` shape). Probability update
    /// is bit-exact with the classic branchy form:
    ///   bit 0: p += (2048 - p) >> 5  ==  p - ((p + 31) >> 5) + 64
    ///   bit 1: p -= p >> 5
    #[inline(always)]
    unsafe fn bit(&mut self, p: *mut Prob, inp: *const u8) -> u32 {
        unsafe { self.normalize(inp) };
        let prob = unsafe { *p } as u32;
        let bound = (self.range >> RC_MODEL_BITS) * prob;
        // t = all ones for bit 0 (code < bound), 0 for bit 1. NB: this must
        // be a real compare — the sign-bit trick `(code - bound) >> 31` is
        // wrong once `range >= 2^31`.
        let t = lt_mask(self.code, bound);
        let nt = !t;
        // bit 0: range = bound;   bit 1: range -= bound, code -= bound
        self.range = bound.wrapping_add(nt & self.range.wrapping_sub(bound << 1));
        self.code = self.code.wrapping_sub(nt & bound);
        unsafe { *p = (prob - ((prob + (t & RC_MODEL_OFFSET)) >> RC_MOVE) + (t & 64)) as Prob };
        nt & 1
    }

    /// Classic branchy bit decode — better for well-predicted bits
    /// (is_match / is_rep / length choice), where a branch costs nothing
    /// and saves the mask arithmetic.
    #[inline(always)]
    unsafe fn bit_br(&mut self, p: *mut Prob, inp: *const u8) -> u32 {
        unsafe { self.normalize(inp) };
        let prob = unsafe { *p } as u32;
        let bound = (self.range >> RC_MODEL_BITS) * prob;
        if self.code < bound {
            self.range = bound;
            unsafe { *p = (prob + ((2048 - prob) >> RC_MOVE)) as Prob };
            0
        } else {
            self.range -= bound;
            self.code -= bound;
            unsafe { *p = (prob - (prob >> RC_MOVE)) as Prob };
            1
        }
    }

    /// Forward bittree over `probs[1 .. 2^num_bits]`.
    #[inline(always)]
    unsafe fn bittree(&mut self, probs: *mut Prob, num_bits: u32, inp: *const u8) -> u32 {
        let mut symbol = 1u32;
        for _ in 0..num_bits {
            let b = unsafe { self.bit(probs.add(symbol as usize), inp) };
            symbol = (symbol << 1) | b;
        }
        symbol - (1 << num_bits)
    }

    /// Reverse bittree (LSB first).
    #[inline(always)]
    unsafe fn bittree_rev(&mut self, probs: *mut Prob, num_bits: u32, inp: *const u8) -> u32 {
        let mut symbol = 1u32;
        let mut result = 0u32;
        for i in 0..num_bits {
            let b = unsafe { self.bit(probs.add(symbol as usize), inp) };
            symbol = (symbol << 1) | b;
            result |= b << i;
        }
        result
    }

    /// Uniform "direct" bits.
    #[inline(always)]
    unsafe fn direct(&mut self, num_bits: u32, inp: *const u8) -> u32 {
        let mut result = 0u32;
        for _ in 0..num_bits {
            unsafe { self.normalize(inp) };
            self.range >>= 1;
            // t = all ones when code < range (bit 0).
            let t = lt_mask(self.code, self.range);
            self.code = self.code.wrapping_sub(self.range & !t);
            result = (result << 1) | (t.wrapping_add(1) & 1);
        }
        result
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
    #[inline(always)]
    unsafe fn decode(&mut self, rc: &mut Rc, pos_state: usize, inp: *const u8) -> u32 {
        unsafe {
            if rc.bit_br(&mut self.choice, inp) == 0 {
                rc.bittree(self.low.get_unchecked_mut(pos_state).as_mut_ptr(), LEN_LOW_BITS, inp)
            } else if rc.bit_br(&mut self.choice2, inp) == 0 {
                LEN_LOW_SYMBOLS as u32
                    + rc.bittree(self.mid.get_unchecked_mut(pos_state).as_mut_ptr(), LEN_MID_BITS, inp)
            } else {
                LEN_LOW_SYMBOLS as u32
                    + LEN_MID_SYMBOLS as u32
                    + rc.bittree(self.high.as_mut_ptr(), LEN_HIGH_BITS, inp)
            }
        }
    }
}

// =========================================================================
// LZMA decoder
// =========================================================================

/// Headroom kept past the write cursor so a full-length match copy
/// (`MATCH_LEN_MAX` = 273) plus the copy kernel's 15-byte overshoot never
/// needs a bounds check.
const OUT_HEADROOM: usize = 320;

/// LZMA stream decoder.  Holds the probability model and the LZ state.
///
/// There is no separate dictionary buffer: the decoder writes straight into
/// the caller's flat `output` Vec and reads back-references from it. The
/// window is `output[dict_start..]`, capped at `dict_size` bytes behind the
/// cursor. Positions used for `lp`/`pb` context are relative to
/// `dict_start`, which mirrors liblzma resetting `dict.pos` on a dict reset.
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

    /// Offset into the output Vec where the current dictionary window
    /// starts (set by the caller on every dict reset).
    pub dict_start: usize,
    /// Maximum back-reference distance.
    pub dict_size: usize,
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
            dict_start: 0,
            dict_size: dict_size.max(1) as usize,
        })
    }

    /// Reset every probability and the state machine.  Does NOT touch the
    /// dictionary window — LZMA2 controls dict reset separately via its
    /// chunk control byte (`dict_start`).
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

    /// Decode the match-distance using the slot/direct/align scheme.
    #[inline(always)]
    unsafe fn decode_distance(&mut self, rc: &mut Rc, len: u32, inp: *const u8) -> u32 {
        let dist_state = get_dist_state(len) as usize;
        unsafe {
            let slot = rc.bittree(
                self.dist_slot.get_unchecked_mut(dist_state).as_mut_ptr(),
                DIST_SLOT_BITS,
                inp,
            );
            if slot < DIST_MODEL_START {
                return slot;
            }
            let num_direct = (slot >> 1) - 1;
            let base: u32 = (2 | (slot & 1)) << num_direct;
            if slot < DIST_MODEL_END {
                // probs[base - slot ..], indexed 1..=2^num_direct by the
                // reverse bittree; base - slot + 2^num_direct <= 128.
                let probs = self.dist_special.as_mut_ptr().add(base as usize - slot as usize);
                base + rc.bittree_rev(probs, num_direct, inp)
            } else {
                let direct_bits = num_direct - ALIGN_BITS;
                let direct = rc.direct(direct_bits, inp) << ALIGN_BITS;
                let align = rc.bittree_rev(self.dist_align.as_mut_ptr(), ALIGN_BITS, inp);
                base + direct + align
            }
        }
    }

    /// Decode bytes from `rd` straight into `output` until either:
    ///   (a) `limit` bytes have been emitted, or
    ///   (b) the end-of-stream marker (distance == u32::MAX) is encountered.
    ///
    /// `rd.input` MUST have enough trailing slack bytes for the range
    /// decoder to refill freely (the hot path has no EOF check).
    ///
    /// Returns `(bytes_emitted, hit_end_marker)`.
    #[inline(never)]
    #[allow(unused_assignments)]
    pub fn decode_into(
        &mut self,
        rd: &mut RangeDecoder,
        limit: usize,
        output: &mut Vec<u8>,
    ) -> io::Result<(usize, bool)> {
        let start = output.len();
        let end = start.saturating_add(limit);
        let dict_start = self.dict_start;
        debug_assert!(dict_start <= start);
        let dict_size = self.dict_size;
        let inp = rd.input.as_ptr();
        let mut rc = Rc { range: rd.range, code: rd.code, pos: rd.pos };
        let lc = self.lc;
        let lp_mask = self.lp_mask as usize;
        let pb_mask = self.pb_mask as usize;
        let lit = self.literal.as_mut_ptr();
        let mut state = self.state;
        let mut reps = self.reps;
        let mut hit_marker = false;

        output.reserve((end - start).min(1 << 22) + OUT_HEADROOM);
        let mut base = output.as_mut_ptr();
        let mut cap = output.capacity();
        let mut pos = start;

        // Only expanded inside the `unsafe` block below.
        macro_rules! fail {
            ($msg:expr) => {{
                output.set_len(pos);
                return Err(io::Error::new(io::ErrorKind::InvalidData, $msg));
            }};
        }

        // SAFETY: every write is at `pos < cap - OUT_HEADROOM` (checked at
        // loop top) and every read is at `>= dict_start >= 0` and `< pos`
        // (distance validated against `pos - dict_start`).
        unsafe {
            while pos < end {
                if pos + OUT_HEADROOM > cap {
                    output.set_len(pos);
                    output.reserve((end - pos).min(1 << 22) + OUT_HEADROOM);
                    base = output.as_mut_ptr();
                    cap = output.capacity();
                }
                let rel = pos - dict_start;
                let pos_state = rel & pb_mask;
                let st = state as usize;

                if rc.bit_br(self.is_match.as_mut_ptr().add(st * POS_STATES_MAX + pos_state), inp) == 0 {
                    // ---- Literal ----
                    let prev = if rel == 0 { 0 } else { *base.add(pos - 1) as u32 };
                    let coder = ((rel & lp_mask) << lc) + (prev >> (8 - lc)) as usize;
                    let probs = lit.add(coder * LITERAL_CODER_SIZE);
                    let mut symbol = 1u32;
                    if state < LIT_STATES {
                        macro_rules! lit_bit {
                            () => {
                                let b = rc.bit(probs.add(symbol as usize), inp);
                                symbol = (symbol << 1) | b;
                            };
                        }
                        lit_bit!(); lit_bit!(); lit_bit!(); lit_bit!();
                        lit_bit!(); lit_bit!(); lit_bit!(); lit_bit!();
                    } else {
                        // Matched literal: the byte at rep0 steers which
                        // sub-tree we descend until the first mismatch,
                        // after which `offset` drops to 0 and the plain
                        // tree is used (liblzma's formulation).
                        let mut match_byte = *base.add(pos - reps[0] as usize - 1) as u32;
                        let mut offset = 0x100u32;
                        macro_rules! mlit_bit {
                            () => {
                                match_byte <<= 1;
                                let match_bit = match_byte & offset;
                                let idx = offset + match_bit + symbol;
                                let b = rc.bit(probs.add(idx as usize), inp);
                                symbol = (symbol << 1) | b;
                                // b == 1: offset &= match_bit; b == 0: offset &= !match_bit
                                offset &= match_bit ^ 0u32.wrapping_sub(b ^ 1);
                            };
                        }
                        mlit_bit!(); mlit_bit!(); mlit_bit!(); mlit_bit!();
                        mlit_bit!(); mlit_bit!(); mlit_bit!(); mlit_bit!();
                    }
                    *base.add(pos) = symbol as u8;
                    pos += 1;
                    state = update_literal(state);
                    continue;
                }

                // ---- Some kind of match ----
                let len: u32;
                if rc.bit_br(self.is_rep.as_mut_ptr().add(st), inp) != 0 {
                    if rel == 0 {
                        fail!("lzma: repeat match with empty dictionary");
                    }
                    if rc.bit_br(self.is_rep0.as_mut_ptr().add(st), inp) == 0 {
                        if rc.bit_br(
                            self.is_rep0_long.as_mut_ptr().add(st * POS_STATES_MAX + pos_state),
                            inp,
                        ) == 0
                        {
                            // Short rep — exactly one byte at rep0.
                            let dist = reps[0] as usize + 1;
                            if dist > rel || dist > dict_size {
                                fail!("lzma: short-rep distance out of range");
                            }
                            *base.add(pos) = *base.add(pos - dist);
                            pos += 1;
                            state = update_short_rep(state);
                            continue;
                        }
                    } else {
                        let dist;
                        if rc.bit_br(self.is_rep1.as_mut_ptr().add(st), inp) == 0 {
                            dist = reps[1];
                        } else {
                            if rc.bit_br(self.is_rep2.as_mut_ptr().add(st), inp) == 0 {
                                dist = reps[2];
                            } else {
                                dist = reps[3];
                                reps[3] = reps[2];
                            }
                            reps[2] = reps[1];
                        }
                        reps[1] = reps[0];
                        reps[0] = dist;
                    }
                    len = MATCH_LEN_MIN + self.rep_len.decode(&mut rc, pos_state, inp);
                    state = update_long_rep(state);
                } else {
                    reps[3] = reps[2];
                    reps[2] = reps[1];
                    reps[1] = reps[0];
                    len = MATCH_LEN_MIN + self.match_len.decode(&mut rc, pos_state, inp);
                    let d = self.decode_distance(&mut rc, len, inp);
                    if d == u32::MAX {
                        hit_marker = true;
                        break;
                    }
                    reps[0] = d;
                    state = update_match(state);
                }

                let dist = reps[0] as usize + 1;
                if dist > rel || dist > dict_size {
                    fail!("lzma: match distance out of range");
                }
                let len = len as usize;
                let copy_len = len.min(end - pos);
                crate::cpu_features::copy_match_unchecked(
                    base.add(pos - dist),
                    base.add(pos),
                    dist,
                    copy_len,
                );
                pos += copy_len;
                if copy_len < len {
                    fail!("lzma: match copy overruns chunk size");
                }
            }
            output.set_len(pos);
        }

        rd.range = rc.range;
        rd.code = rc.code;
        rd.pos = rc.pos;
        self.state = state;
        self.reps = reps;
        Ok((pos - start, hit_marker))
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
