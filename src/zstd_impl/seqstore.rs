//! Sequence store + the small LZ helpers shared by the match finders
//! (mirrors `SeqStore_t`, `ZSTD_storeSeq`, `ZSTD_count`, `ZSTD_hashPtr`).
//!
//! Every `unsafe` the match finders need is funnelled through the helpers in
//! this file, so there are exactly three kinds to audit:
//!
//! 1. unchecked slice reads — [`byte`], [`read16`], [`read32`], [`read64`],
//!    [`count`], [`hash_at`] (precondition: the bytes read are inside the
//!    slice);
//! 2. unchecked table indexing — [`tbl_get`], [`tbl_set`] (precondition:
//!    `idx < tbl.len()`);
//! 3. the literal wildcopy inside [`SeqStore::store_seq`], which is `unsafe`
//!    only internally (its branches establish what the copy needs).
//!
//! Each helper states its precondition and checks it with `debug_assert!`.

/// One parsed sequence. `off_base` is the decoder-facing offset value:
/// `offset + 3` for a real offset, `1..=3` for a repeat code (as in C's
/// `OFFSET_TO_OFFBASE` / `REPCODE_TO_OFFBASE`).
#[derive(Clone, Copy, Default, Debug)]
#[repr(C)]
pub struct SeqDef {
    pub off_base: u32,
    pub lit_len: u32,
    /// match length - MINMATCH (3)
    pub ml_base: u32,
}

pub const MIN_MATCH: usize = 3;
pub const REPCODE1_TO_OFFBASE: u32 = 1;
/// Literals are copied 32 bytes at a time past their end; the source must
/// have this much slack or the safe copy is used.
pub const WILDCOPY_OVERLENGTH: usize = 32;

#[inline(always)]
pub fn offset_to_offbase(offset: u32) -> u32 {
    offset + 3
}

/// Per-block sequence store. Literals go into `lit`, sequences into `seqs`.
pub struct SeqStore {
    pub lit: Vec<u8>,
    pub seqs: Vec<SeqDef>,
    /// Longest literal run seen (to size the LL code path).
    pub long_lit: bool,
}

impl SeqStore {
    pub fn new() -> Self {
        SeqStore { lit: Vec::new(), seqs: Vec::new(), long_lit: false }
    }

    /// Prepare for a block of `block_size` bytes: room for every literal plus
    /// the wildcopy slack, and one sequence per 4 bytes (min match is 4 for
    /// every parser). With this reserve neither `push` below ever grows.
    pub fn reset(&mut self, block_size: usize) {
        self.lit.clear();
        self.lit.reserve(block_size + WILDCOPY_OVERLENGTH);
        self.seqs.clear();
        self.seqs.reserve(block_size / 4 + 2);
        self.long_lit = false;
    }

    /// `ZSTD_storeSeq`: copy the literals `base[lit_start..lit_start +
    /// lit_len]` and append the sequence.
    ///
    /// # Safety
    /// `lit_start + lit_len <= base.len()`, and `reset` was called with a
    /// block size covering every literal and sequence stored since (so the
    /// `lit` capacity has `WILDCOPY_OVERLENGTH` spare bytes and `seqs` has
    /// a free slot). Checking the capacities at runtime instead measured
    /// ~3% on zstd compress, so they are `debug_assert`s.
    #[inline(always)]
    pub unsafe fn store_seq(&mut self, lit_start: usize, lit_len: usize, base: &[u8], off_base: u32, match_len: usize) {
        let len = self.lit.len();
        let lit_end = lit_start + lit_len;
        debug_assert!(lit_end <= base.len());
        debug_assert!(len + lit_len + WILDCOPY_OVERLENGTH <= self.lit.capacity(), "SeqStore::reset undersized");
        // Common case: the source has slack — copy 16 first, then 32 at a
        // time, overshooting the real length (into `lit`'s spare capacity).
        if lit_end + WILDCOPY_OVERLENGTH <= base.len() {
            // SAFETY: the source has `WILDCOPY_OVERLENGTH` readable bytes
            // past the literals (checked) and `lit` has as many spare bytes
            // of capacity past `len` (contract), so every 16/32-byte block
            // the copies touch is inside its allocation, the regions belong
            // to different allocations, and `set_len` only exposes bytes the
            // copy wrote.
            unsafe {
                let src = base.as_ptr().add(lit_start);
                let dst = self.lit.as_mut_ptr().add(len);
                copy16(src, dst);
                if lit_len > 16 {
                    wildcopy32(src.add(16), dst.add(16), lit_len - 16);
                }
                self.lit.set_len(len + lit_len);
            }
        } else {
            // SAFETY: `lit_end <= base.len()` (contract) and `lit` has
            // `lit_len` spare bytes (contract); distinct allocations.
            unsafe {
                core::ptr::copy_nonoverlapping(base.as_ptr().add(lit_start), self.lit.as_mut_ptr().add(len), lit_len);
                self.lit.set_len(len + lit_len);
            }
        }

        debug_assert!(match_len >= MIN_MATCH);
        debug_assert!(self.seqs.len() < self.seqs.capacity(), "SeqStore::reset undersized");
        let n = self.seqs.len();
        // SAFETY: `seqs` has a free slot (contract); the slot is written
        // before `set_len` exposes it.
        unsafe {
            core::ptr::write(
                self.seqs.as_mut_ptr().add(n),
                SeqDef { off_base, lit_len: lit_len as u32, ml_base: (match_len - MIN_MATCH) as u32 },
            );
            self.seqs.set_len(n + 1);
        }
    }

    /// Append the block's trailing literals.
    pub fn store_last_literals(&mut self, literals: &[u8]) {
        self.lit.extend_from_slice(literals);
    }
}

// ---------------------------------------------------------------------------
// Wildcopy (private; only `store_seq` uses it)
// ---------------------------------------------------------------------------

/// # Safety
/// 16 bytes readable at `src` and writable at `dst`, no overlap.
#[inline(always)]
unsafe fn copy16(src: *const u8, dst: *mut u8) {
    // SAFETY: per the contract above.
    unsafe {
        let v = core::ptr::read_unaligned(src as *const [u8; 16]);
        core::ptr::write_unaligned(dst as *mut [u8; 16], v);
    }
}

/// Copy in 32-byte steps, overshooting by up to 31 bytes.
///
/// # Safety
/// `len.max(1).next_multiple_of(32)` bytes readable at `src` and writable
/// at `dst`, no overlap.
#[inline(always)]
unsafe fn wildcopy32(mut src: *const u8, mut dst: *mut u8, len: usize) {
    // SAFETY: per the contract above; every iteration copies one 32-byte
    // step and stops once `dst` reaches or passes `end`.
    unsafe {
        let end = dst.add(len);
        loop {
            copy16(src, dst);
            copy16(src.add(16), dst.add(16));
            src = src.add(32);
            dst = dst.add(32);
            if dst >= end {
                return;
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Unchecked slice reads
// ---------------------------------------------------------------------------

/// # Safety
/// `pos < buf.len()`.
#[inline(always)]
pub unsafe fn byte(buf: &[u8], pos: usize) -> u8 {
    debug_assert!(pos < buf.len());
    // SAFETY: per the contract above.
    unsafe { *buf.get_unchecked(pos) }
}

/// Little-endian `u16` at `pos`.
///
/// # Safety
/// `pos + 2 <= buf.len()`.
#[inline(always)]
pub unsafe fn read16(buf: &[u8], pos: usize) -> u16 {
    debug_assert!(pos + 2 <= buf.len());
    // SAFETY: per the contract above; unaligned load.
    unsafe { u16::from_le(core::ptr::read_unaligned(buf.as_ptr().add(pos) as *const u16)) }
}

/// Little-endian `u32` at `pos`.
///
/// # Safety
/// `pos + 4 <= buf.len()`.
#[inline(always)]
pub unsafe fn read32(buf: &[u8], pos: usize) -> u32 {
    debug_assert!(pos + 4 <= buf.len());
    // SAFETY: per the contract above; unaligned load.
    unsafe { u32::from_le(core::ptr::read_unaligned(buf.as_ptr().add(pos) as *const u32)) }
}

/// Little-endian `u64` at `pos`.
///
/// # Safety
/// `pos + 8 <= buf.len()`.
#[inline(always)]
pub unsafe fn read64(buf: &[u8], pos: usize) -> u64 {
    debug_assert!(pos + 8 <= buf.len());
    // SAFETY: per the contract above; unaligned load.
    unsafe { u64::from_le(core::ptr::read_unaligned(buf.as_ptr().add(pos) as *const u64)) }
}

/// `ZSTD_count`: number of equal bytes at `buf[p_in..]` / `buf[p_match..]`,
/// reading no further than `limit` (exclusive) on the `p_in` side.
///
/// # Safety
/// `p_match <= p_in <= limit <= buf.len()`.
#[inline(always)]
pub unsafe fn count(buf: &[u8], p_in: usize, p_match: usize, limit: usize) -> usize {
    debug_assert!(p_match <= p_in && p_in <= limit && limit <= buf.len());
    let start = p_in;
    let mut p_in = p_in;
    let mut p_match = p_match;
    // Hoisted once, as C does (`pInLimit - 7`): the loop condition stays a
    // single compare.
    let loop_limit = limit.saturating_sub(7);
    // SAFETY: every read below is guarded by `p_in + N <= limit` (as
    // `p_in < limit - (N - 1)`), and `p_match <= p_in` keeps the match side
    // in range too.
    unsafe {
        if p_in < loop_limit {
            let diff = read64(buf, p_match) ^ read64(buf, p_in);
            if diff != 0 {
                return (diff.trailing_zeros() >> 3) as usize;
            }
            p_in += 8;
            p_match += 8;
            while p_in < loop_limit {
                let diff = read64(buf, p_match) ^ read64(buf, p_in);
                if diff == 0 {
                    p_in += 8;
                    p_match += 8;
                    continue;
                }
                p_in += (diff.trailing_zeros() >> 3) as usize;
                return p_in - start;
            }
        }
        if p_in + 3 < limit && read32(buf, p_match) == read32(buf, p_in) {
            p_in += 4;
            p_match += 4;
        }
        if p_in + 1 < limit && read16(buf, p_match) == read16(buf, p_in) {
            p_in += 2;
            p_match += 2;
        }
        if p_in < limit && byte(buf, p_match) == byte(buf, p_in) {
            p_in += 1;
        }
    }
    p_in - start
}

// ---------------------------------------------------------------------------
// Hashes (ZSTD_hashPtr family)
// ---------------------------------------------------------------------------

const PRIME4: u32 = 2654435761;
const PRIME5: u64 = 889523592379;
const PRIME6: u64 = 227718039650203;
const PRIME7: u64 = 58295818150454627;
const PRIME8: u64 = 0xCF1BBCDCB7A56463;

/// Hash `MLS` bytes at `buf[pos..]` to `h_bits` bits. `MLS` in 4..=8.
///
/// # Safety
/// `pos + 8 <= buf.len()` when `MLS > 4`, `pos + 4 <= buf.len()` otherwise.
#[inline(always)]
pub unsafe fn hash_at<const MLS: u32>(buf: &[u8], pos: usize, h_bits: u32) -> usize {
    // SAFETY: per the contract above.
    unsafe {
        match MLS {
            4 => ((read32(buf, pos).wrapping_mul(PRIME4)) >> (32 - h_bits)) as usize,
            5 => (((read64(buf, pos) << (64 - 40)).wrapping_mul(PRIME5)) >> (64 - h_bits)) as usize,
            6 => (((read64(buf, pos) << (64 - 48)).wrapping_mul(PRIME6)) >> (64 - h_bits)) as usize,
            7 => (((read64(buf, pos) << (64 - 56)).wrapping_mul(PRIME7)) >> (64 - h_bits)) as usize,
            _ => ((read64(buf, pos).wrapping_mul(PRIME8)) >> (64 - h_bits)) as usize,
        }
    }
}

// ---------------------------------------------------------------------------
// Unchecked table indexing
// ---------------------------------------------------------------------------

/// # Safety
/// `idx < tbl.len()`.
#[inline(always)]
pub unsafe fn tbl_get<T: Copy>(tbl: &[T], idx: usize) -> T {
    debug_assert!(idx < tbl.len());
    // SAFETY: per the contract above.
    unsafe { *tbl.get_unchecked(idx) }
}

/// # Safety
/// `idx < tbl.len()`.
#[inline(always)]
pub unsafe fn tbl_set<T>(tbl: &mut [T], idx: usize, v: T) {
    debug_assert!(idx < tbl.len());
    // SAFETY: per the contract above.
    unsafe { *tbl.get_unchecked_mut(idx) = v }
}

#[inline(always)]
pub fn highbit32(v: u32) -> u32 {
    debug_assert!(v != 0);
    31 - v.leading_zeros()
}
