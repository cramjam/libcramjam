//! Sequence store + the small LZ helpers shared by the match finders
//! (mirrors `SeqStore_t`, `ZSTD_storeSeq`, `ZSTD_count`, `ZSTD_hashPtr`).

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

/// Per-block sequence store. Literals go into `lit` (raw-pointer writes with
/// slack for the wildcopy), sequences into `seqs`.
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
    /// every parser).
    pub fn reset(&mut self, block_size: usize) {
        self.lit.clear();
        self.lit.reserve(block_size + WILDCOPY_OVERLENGTH);
        self.seqs.clear();
        self.seqs.reserve(block_size / 4 + 2);
        self.long_lit = false;
    }

    /// `ZSTD_storeSeq`: copy `lit_len` literals from `literals` and append the
    /// sequence.
    ///
    /// # Safety
    /// `[literals, literals + lit_len)` must be readable and, when
    /// `literals + lit_len + WILDCOPY_OVERLENGTH <= lit_limit`, the whole
    /// wildcopy window must be readable too. `reset` must have been called
    /// with a block size covering all literals stored since.
    #[inline(always)]
    pub unsafe fn store_seq(&mut self, lit_len: usize, literals: *const u8, lit_limit: *const u8, off_base: u32, match_len: usize) {
        let len = self.lit.len();
        debug_assert!(len + lit_len + WILDCOPY_OVERLENGTH <= self.lit.capacity());
        let dst = unsafe { self.lit.as_mut_ptr().add(len) };
        let lit_end = unsafe { literals.add(lit_len) };
        if unsafe { lit_end.add(WILDCOPY_OVERLENGTH) } <= lit_limit {
            // Common case: literals are short, copy 16 first, then wildcopy.
            unsafe {
                copy16(literals, dst);
                if lit_len > 16 {
                    wildcopy32(literals.add(16), dst.add(16), lit_len - 16);
                }
            }
        } else {
            unsafe { core::ptr::copy_nonoverlapping(literals, dst, lit_len) };
        }
        unsafe { self.lit.set_len(len + lit_len) };

        debug_assert!(match_len >= MIN_MATCH);
        debug_assert!(self.seqs.len() < self.seqs.capacity());
        let n = self.seqs.len();
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
// Memory helpers
// ---------------------------------------------------------------------------

#[inline(always)]
pub unsafe fn read16(p: *const u8) -> u16 {
    unsafe { u16::from_le(core::ptr::read_unaligned(p as *const u16)) }
}
#[inline(always)]
pub unsafe fn read32(p: *const u8) -> u32 {
    unsafe { u32::from_le(core::ptr::read_unaligned(p as *const u32)) }
}
#[inline(always)]
pub unsafe fn read64(p: *const u8) -> u64 {
    unsafe { u64::from_le(core::ptr::read_unaligned(p as *const u64)) }
}

#[inline(always)]
pub unsafe fn copy16(src: *const u8, dst: *mut u8) {
    unsafe {
        let v = core::ptr::read_unaligned(src as *const [u8; 16]);
        core::ptr::write_unaligned(dst as *mut [u8; 16], v);
    }
}

/// Copy in 32-byte steps, overshooting by up to 31 bytes.
#[inline(always)]
pub unsafe fn wildcopy32(mut src: *const u8, mut dst: *mut u8, len: usize) {
    let end = unsafe { dst.add(len) };
    loop {
        unsafe {
            copy16(src, dst);
            copy16(src.add(16), dst.add(16));
            src = src.add(32);
            dst = dst.add(32);
        }
        if dst >= end {
            return;
        }
    }
}

/// `ZSTD_count`: number of equal bytes at `p_in` / `p_match`, reading no
/// further than `limit` (exclusive) on the `p_in` side.
///
/// # Safety
/// `[p_in, limit)` and the matching range at `p_match` must be readable.
#[inline(always)]
pub unsafe fn count(p_in: *const u8, p_match: *const u8, limit: *const u8) -> usize {
    let start = p_in;
    let mut p_in = p_in;
    let mut p_match = p_match;
    let loop_limit = unsafe { limit.sub(7) };
    if p_in < loop_limit {
        let diff = unsafe { read64(p_match) ^ read64(p_in) };
        if diff != 0 {
            return (diff.trailing_zeros() >> 3) as usize;
        }
        p_in = unsafe { p_in.add(8) };
        p_match = unsafe { p_match.add(8) };
        while p_in < loop_limit {
            let diff = unsafe { read64(p_match) ^ read64(p_in) };
            if diff == 0 {
                p_in = unsafe { p_in.add(8) };
                p_match = unsafe { p_match.add(8) };
                continue;
            }
            p_in = unsafe { p_in.add((diff.trailing_zeros() >> 3) as usize) };
            return unsafe { p_in.offset_from(start) } as usize;
        }
    }
    unsafe {
        if p_in < limit.sub(3) && read32(p_match) == read32(p_in) {
            p_in = p_in.add(4);
            p_match = p_match.add(4);
        }
        if p_in < limit.sub(1) && read16(p_match) == read16(p_in) {
            p_in = p_in.add(2);
            p_match = p_match.add(2);
        }
        if p_in < limit && *p_match == *p_in {
            p_in = p_in.add(1);
        }
        p_in.offset_from(start) as usize
    }
}

// ---------------------------------------------------------------------------
// Hashes (ZSTD_hashPtr family)
// ---------------------------------------------------------------------------

const PRIME4: u32 = 2654435761;
const PRIME5: u64 = 889523592379;
const PRIME6: u64 = 227718039650203;
const PRIME7: u64 = 58295818150454627;
const PRIME8: u64 = 0xCF1BBCDCB7A56463;

/// Hash `MLS` bytes at `p` to `h_bits` bits. `MLS` in 4..=8.
///
/// # Safety
/// 8 bytes must be readable at `p` when `MLS > 4`, 4 otherwise.
#[inline(always)]
pub unsafe fn hash_ptr<const MLS: u32>(p: *const u8, h_bits: u32) -> usize {
    match MLS {
        4 => ((unsafe { read32(p) }.wrapping_mul(PRIME4)) >> (32 - h_bits)) as usize,
        5 => (((unsafe { read64(p) } << (64 - 40)).wrapping_mul(PRIME5)) >> (64 - h_bits)) as usize,
        6 => (((unsafe { read64(p) } << (64 - 48)).wrapping_mul(PRIME6)) >> (64 - h_bits)) as usize,
        7 => (((unsafe { read64(p) } << (64 - 56)).wrapping_mul(PRIME7)) >> (64 - h_bits)) as usize,
        _ => ((unsafe { read64(p) }.wrapping_mul(PRIME8)) >> (64 - h_bits)) as usize,
    }
}

#[inline(always)]
pub fn highbit32(v: u32) -> u32 {
    debug_assert!(v != 0);
    31 - v.leading_zeros()
}
