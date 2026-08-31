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

use super::range_coder::{prob_init, prob_reset_slice, Prob, RangeDecoder};

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
    /// One past the last valid compressed byte (chunk length). Reads at or
    /// beyond this saturate to 0 and set `overrun` instead of touching
    /// memory — a corrupt stream (e.g. tiny compressed_size but huge
    /// uncompressed_size) must never read past the input allocation.
    end: usize,
    overrun: bool,
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
            let byte = if self.pos < self.end {
                unsafe { *inp.add(self.pos) }
            } else {
                self.overrun = true;
                0
            };
            self.code = (self.code << 8) | byte as u32;
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
        let mut rc = Rc { range: rd.range, code: rd.code, pos: rd.pos, end: rd.end, overrun: false };
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

        if rc.overrun {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "lzma: range decoder read past end of compressed data",
            ));
        }
        rd.range = rc.range;
        rd.code = rc.code;
        rd.pos = rc.pos;
        self.state = state;
        self.reps = reps;
        Ok((pos - start, hit_marker))
    }
}
