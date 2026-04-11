//! Runtime CPU feature detection + SIMD wildcopy kernels.
//!
//! Feature queries are cached in a relaxed `AtomicU32`; the hot path is a
//! single relaxed load. `std::is_x86_feature_detected!` runs at most once per
//! process per feature. aarch64 NEON is a baseline for `aarch64-unknown-*`
//! targets so the query is a no-op `true` there; other targets return `false`.
//!
//! The `wildcopy_chunks::<N>` kernel is the shared workhorse used by the lz4
//! and zstd decoders to execute a match (back-reference) copy. It copies
//! `length` bytes in `N`-byte chunks and may write up to `N-1` extra bytes
//! past the end. Callers MUST have reserved headroom in the destination for
//! the overshoot.
//!
//! Runtime feature dispatch lets us keep the published wheel portable while
//! still generating AVX2 code for modern CPUs. Locally, `.cargo/config.toml`
//! sets `target-cpu=x86-64-v3` which enables AVX2 statically — on published
//! builds the runtime check picks up the real CPU.

#![allow(dead_code)]

#[cfg(target_arch = "x86_64")]
use std::sync::atomic::{AtomicU32, Ordering};

// ---------------------------------------------------------------------------
// Feature detection
// ---------------------------------------------------------------------------

/// `true` if the host supports AVX2.
///
/// Cached after first call. Returns `false` on non-x86_64 targets.
#[inline(always)]
pub fn has_avx2() -> bool {
    #[cfg(target_arch = "x86_64")]
    {
        // Sentinel: `u32::MAX` = not yet detected.
        static CACHE: AtomicU32 = AtomicU32::new(u32::MAX);
        let cached = CACHE.load(Ordering::Relaxed);
        if cached != u32::MAX {
            return cached != 0;
        }
        let detected = std::is_x86_feature_detected!("avx2");
        CACHE.store(detected as u32, Ordering::Relaxed);
        detected
    }
    #[cfg(not(target_arch = "x86_64"))]
    {
        false
    }
}

/// `true` if the host supports 128-bit SIMD (SSE2 on x86_64 is baseline, NEON
/// on aarch64 is baseline for most targets). This is effectively a compile-
/// time query except on aarch64 without `target_feature = "neon"`.
#[inline(always)]
pub fn has_simd128() -> bool {
    #[cfg(target_arch = "x86_64")]
    {
        true
    }
    #[cfg(target_arch = "aarch64")]
    {
        #[cfg(target_feature = "neon")]
        {
            true
        }
        #[cfg(not(target_feature = "neon"))]
        {
            std::arch::is_aarch64_feature_detected!("neon")
        }
    }
    #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
    {
        false
    }
}

// ---------------------------------------------------------------------------
// Wildcopy kernel
// ---------------------------------------------------------------------------

/// Copy `length` bytes from `src` into `dst` in fixed `N`-byte chunks using
/// unaligned loads and stores. May write up to `N - 1` bytes past the end of
/// the requested region.
///
/// For `N = 16` on x86_64 this lowers to `movdqu` pairs; on aarch64 to
/// unaligned `ldr q?`/`str q?`. For `N = 32` on a target with AVX2 enabled it
/// lowers to `vmovdqu` pairs. The const-generic `N` lets the compiler
/// specialize the loop body for a specific chunk size.
///
/// # Safety
///
/// * `[src, src + length + N)` must be a valid read region.
/// * `[dst, dst + length + N)` must be a valid write region.
/// * For match (back-reference) copies the regions MAY overlap, but only if
///   `dst >= src + N`, i.e. every chunk reads from bytes that were fully
///   written by earlier chunks (or by the caller before the loop started).
///   For `N = 16` this means callers must guard `offset >= 16` before using
///   this helper on a self-referential match.
#[inline(always)]
pub unsafe fn wildcopy_chunks<const N: usize>(
    mut src: *const u8,
    mut dst: *mut u8,
    length: usize,
) {
    if length == 0 {
        return;
    }
    // SAFETY: caller guarantees `src + length + N` is valid to read.
    let end = unsafe { src.add(length) };
    loop {
        // Const-N unaligned array load/store. LLVM reliably lowers these to
        // a single wide move of the right width (movdqu/vmovdqu/ldp/stp).
        let chunk: [u8; N] = unsafe { core::ptr::read_unaligned(src.cast::<[u8; N]>()) };
        unsafe { core::ptr::write_unaligned(dst.cast::<[u8; N]>(), chunk) };
        src = unsafe { src.add(N) };
        dst = unsafe { dst.add(N) };
        if src >= end {
            return;
        }
    }
}

/// Match-execution kernel shared by lz4 and zstd decoders.
///
/// Copies `match_len` bytes from `src` (points at `dst - offset`) into `dst`.
/// Dispatches on three cases:
///
/// 1. `offset >= 16`: non-overlapping 16-byte wildcopy.
/// 2. `offset == 1`: single-byte RLE via `write_bytes`.
/// 3. `2 <= offset < 16`: overlapping copy in `offset`-sized chunks so the
///    pattern propagates. Falls through to a tail copy.
///
/// # Safety
///
/// * `src` must point `offset` bytes before `dst`.
/// * `[src, dst + match_len + 16)` must be a valid in-allocation region.
/// * Caller is responsible for advancing the destination cursor (e.g. via
///   `Vec::set_len`) after calling this function.
#[inline(always)]
pub unsafe fn copy_match_unchecked(src: *const u8, dst: *mut u8, offset: usize, match_len: usize) {
    if offset >= 16 {
        unsafe { wildcopy_chunks::<16>(src, dst, match_len) };
    } else if offset == 1 {
        // RLE — very common for runs of padding bytes, whitespace, zeros.
        unsafe { core::ptr::write_bytes(dst, *src, match_len) };
    } else {
        // Overlapping: copy `offset` bytes at a time so each chunk reads
        // already-written data and the pattern propagates correctly.
        let mut w = 0usize;
        while w + offset <= match_len {
            unsafe { core::ptr::copy_nonoverlapping(src.add(w), dst.add(w), offset) };
            w += offset;
        }
        if w < match_len {
            unsafe { core::ptr::copy_nonoverlapping(src.add(w), dst.add(w), match_len - w) };
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn wildcopy_basic() {
        let src = b"Hello, world! This is a test string. More bytes here.";
        let mut dst = vec![0u8; 64];
        unsafe {
            wildcopy_chunks::<16>(src.as_ptr(), dst.as_mut_ptr(), src.len());
        }
        assert_eq!(&dst[..src.len()], src);
    }

    #[test]
    fn copy_match_non_overlapping() {
        let mut buf = vec![0u8; 128];
        buf[..20].copy_from_slice(b"abcdefghijklmnopqrst");
        unsafe {
            let p = buf.as_mut_ptr();
            copy_match_unchecked(p, p.add(20), 20, 15);
        }
        assert_eq!(&buf[20..35], b"abcdefghijklmno");
    }

    #[test]
    fn copy_match_rle_one() {
        let mut buf = vec![0u8; 64];
        buf[0] = b'Z';
        unsafe {
            let p = buf.as_mut_ptr();
            copy_match_unchecked(p, p.add(1), 1, 10);
        }
        assert_eq!(&buf[..11], b"ZZZZZZZZZZZ");
    }

    #[test]
    fn copy_match_overlapping_small() {
        let mut buf = vec![0u8; 64];
        buf[..4].copy_from_slice(b"ABCD");
        unsafe {
            let p = buf.as_mut_ptr();
            copy_match_unchecked(p, p.add(4), 4, 10);
        }
        assert_eq!(&buf[..14], b"ABCDABCDABCDAB");
    }
}
