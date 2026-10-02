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
#![deny(clippy::undocumented_unsafe_blocks)]

#[cfg(all(target_arch = "x86_64", not(miri)))]
use std::sync::atomic::{AtomicU32, Ordering};

// ---------------------------------------------------------------------------
// Feature detection
// ---------------------------------------------------------------------------

/// `true` if the host supports AVX2.
///
/// Cached after first call. Returns `false` on non-x86_64 targets.
#[inline(always)]
pub fn has_avx2() -> bool {
    // Miri has no SIMD/asm support: force the portable paths.
    #[cfg(all(target_arch = "x86_64", not(miri)))]
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
    #[cfg(any(not(target_arch = "x86_64"), miri))]
    {
        false
    }
}

/// `true` if the host supports BMI2 (`shlx`/`shrx`/`bzhi`). Cached like
/// [`has_avx2`]. Returns `false` on non-x86_64 targets.
#[inline(always)]
pub fn has_bmi2() -> bool {
    #[cfg(all(target_arch = "x86_64", not(miri)))]
    {
        static CACHE: AtomicU32 = AtomicU32::new(u32::MAX);
        let cached = CACHE.load(Ordering::Relaxed);
        if cached != u32::MAX {
            return cached != 0;
        }
        let detected = std::is_x86_feature_detected!("bmi2");
        CACHE.store(detected as u32, Ordering::Relaxed);
        detected
    }
    #[cfg(any(not(target_arch = "x86_64"), miri))]
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
/// unaligned loads and stores. Always copies at least one full chunk (even
/// when `length == 0`) and may write up to `N - 1` bytes past the end of the
/// requested region — the copy-first-then-test shape is what makes the
/// common `length <= N` case a single load/store pair with one predictable
/// branch (mirrors `ZSTD_wildcopy` / `LZ4_wildCopy`).
///
/// # Safety
///
/// * `[src, src + max(length, N) + N)` must be a valid read region.
/// * `[dst, dst + max(length, N) + N)` must be a valid write region.
/// * For match (back-reference) copies the regions MAY overlap, but only if
///   `dst >= src + N`, i.e. every chunk reads from bytes that were fully
///   written by earlier chunks (or by the caller before the loop started).
#[inline(always)]
pub unsafe fn wildcopy_chunks<const N: usize>(
    mut src: *const u8,
    mut dst: *mut u8,
    length: usize,
) {
    // SAFETY: per the contract each N-byte step stays inside the
    // `max(length, N) + N` windows, and overlapping regions satisfy
    // `dst >= src + N` so every read sees already-written bytes.
    unsafe {
        let end = dst.add(length);
        loop {
            let chunk: [u8; N] = core::ptr::read_unaligned(src.cast::<[u8; N]>());
            core::ptr::write_unaligned(dst.cast::<[u8; N]>(), chunk);
            src = src.add(N);
            dst = dst.add(N);
            if dst >= end {
                return;
            }
        }
    }
}

/// # Safety
/// 8 bytes readable at `src` and writable at `dst`.
#[inline(always)]
unsafe fn copy8(src: *const u8, dst: *mut u8) {
    // SAFETY: per the contract.
    unsafe {
        let v = core::ptr::read_unaligned(src as *const u64);
        core::ptr::write_unaligned(dst as *mut u64, v);
    }
}

/// After the first 8 bytes of an overlapping copy with `offset < 8`, shift
/// `src` so that the effective offset becomes >= 8 (lz4's `inc32table` /
/// `dec64table`, zstd's `ZSTD_overlapCopy8`).
const INC32: [usize; 8] = [0, 1, 2, 1, 0, 4, 4, 4];
const DEC64: [isize; 8] = [0, 0, 0, -1, -4, 1, 2, 3];

/// zstd's match copy (`ZSTD_execSequence`): like [`copy_match_unchecked`]
/// but the non-overlapping path copies 16 bytes then continues in 32-byte
/// steps (two ordered 16-byte moves, so `offset >= 16` stays correct) —
/// half the loop iterations, and therefore half the loop-exit branch
/// mispredicts, on matches longer than 16 bytes.
///
/// # Safety
/// As [`copy_match_unchecked`], but may write up to **31** bytes past
/// `dst + match_len` (callers need 32 bytes of headroom).
#[inline(always)]
pub unsafe fn copy_match_unchecked_32(src: *const u8, dst: *mut u8, offset: usize, match_len: usize) {
    if offset >= 16 {
        // SAFETY: `offset >= 16` makes each ordered 16-byte move read bytes
        // written before it; the caller provides `match_len + 32` writable
        // bytes, and `offset >= match_len` makes the memcpy non-overlapping.
        unsafe {
            let a: [u8; 16] = core::ptr::read_unaligned(src.cast());
            core::ptr::write_unaligned(dst.cast::<[u8; 16]>(), a);
            if match_len <= 16 {
                return;
            }
            if match_len > 64 && offset >= match_len {
                core::ptr::copy_nonoverlapping(src.add(16), dst.add(16), match_len - 16);
                return;
            }
            let end = dst.add(match_len);
            let mut s = src.add(16);
            let mut d = dst.add(16);
            loop {
                let a: [u8; 16] = core::ptr::read_unaligned(s.cast());
                core::ptr::write_unaligned(d.cast::<[u8; 16]>(), a);
                let b: [u8; 16] = core::ptr::read_unaligned(s.add(16).cast());
                core::ptr::write_unaligned(d.add(16).cast::<[u8; 16]>(), b);
                s = s.add(32);
                d = d.add(32);
                if d >= end {
                    return;
                }
            }
        }
    }
    // SAFETY: same contract, 16 bytes of headroom is a subset of 32.
    unsafe { copy_match_unchecked(src, dst, offset, match_len) }
}

/// Match-execution kernel shared by lz4 and zstd decoders.
///
/// Copies `match_len` bytes from `src` (points at `dst - offset`) into `dst`.
///
/// 1. `offset >= 16`: non-overlapping 16-byte wildcopy.
/// 2. `offset < 16`: fix up the first 8 bytes so the effective offset is
///    >= 8, then stream 8-byte chunks (each reads bytes already written).
///
/// May write up to 15 bytes past `dst + match_len`.
///
/// # Safety
///
/// * `src` must point `offset` bytes before `dst`, `offset >= 1`.
/// * `[src, dst + max(match_len, 16) + 16)` must be a valid in-allocation region.
/// * Caller is responsible for advancing the destination cursor (e.g. via
///   `Vec::set_len`) after calling this function.
#[inline(always)]
pub unsafe fn copy_match_unchecked(src: *const u8, dst: *mut u8, offset: usize, match_len: usize) {
    if offset >= 16 {
        // Long, non-overlapping matches (nci-style data): libc memcpy moves
        // 32-64 bytes/cycle with AVX/`rep movsb`, vs 16 per iteration here.
        if match_len > 64 && offset >= match_len {
            // SAFETY: `offset >= match_len` means the regions are disjoint.
            unsafe { core::ptr::copy_nonoverlapping(src, dst, match_len) };
            return;
        }
        // SAFETY: `offset >= 16 == N` satisfies the kernel's overlap rule;
        // the caller provides the `+ 16` headroom.
        unsafe { wildcopy_chunks::<16>(src, dst, match_len) };
        return;
    }
    let mut src = src;
    let mut dst = dst;
    // SAFETY: `offset >= 1`: the first 8 bytes are produced byte-wise (or
    // via the inc/dec tables) so that afterwards `dst - src >= 8`, and the
    // 8-byte loop then only reads written bytes; writes stay within the
    // caller's `max(match_len, 16) + 16` window.
    unsafe {
        let end = dst.add(match_len);
        if offset < 8 {
            *dst = *src;
            *dst.add(1) = *src.add(1);
            *dst.add(2) = *src.add(2);
            *dst.add(3) = *src.add(3);
            src = src.add(*INC32.get_unchecked(offset));
            let v = core::ptr::read_unaligned(src as *const u32);
            core::ptr::write_unaligned(dst.add(4) as *mut u32, v);
            src = src.offset(-*DEC64.get_unchecked(offset));
        } else {
            copy8(src, dst);
            src = src.add(8);
        }
        dst = dst.add(8);
        while dst < end {
            copy8(src, dst);
            src = src.add(8);
            dst = dst.add(8);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn wildcopy_basic() {
        let text = b"Hello, world! This is a test string. More bytes here.";
        // The kernel over-reads up to N bytes: give the source that slack.
        let mut src = text.to_vec();
        src.resize(text.len() + 32, 0);
        let mut dst = vec![0u8; 128];
        // SAFETY: `src` has `len + 32` readable bytes, `dst` 128 writable.
        unsafe {
            wildcopy_chunks::<16>(src.as_ptr(), dst.as_mut_ptr(), text.len());
        }
        assert_eq!(&dst[..text.len()], text);
    }

    #[test]
    fn copy_match_non_overlapping() {
        let mut buf = vec![0u8; 128];
        buf[..20].copy_from_slice(b"abcdefghijklmnopqrst");
        // SAFETY: 128-byte buffer, copy touches at most `[0, 20 + 15 + 16)`.
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
        // SAFETY: 64-byte buffer, copy touches at most `[0, 1 + 10 + 16)`.
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
        // SAFETY: 64-byte buffer, copy touches at most `[0, 4 + 10 + 16)`.
        unsafe {
            let p = buf.as_mut_ptr();
            copy_match_unchecked(p, p.add(4), 4, 10);
        }
        assert_eq!(&buf[..14], b"ABCDABCDABCDAB");
    }
}
