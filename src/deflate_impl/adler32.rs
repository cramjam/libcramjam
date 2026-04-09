//! Adler-32 as specified in RFC 1950
//!
//! Delegates to `simd-adler32` which uses SIMD acceleration (SSE2/AVX2/NEON)
//! when available, falling back to a fast software implementation.

/// Compute Adler-32 of a byte slice.
#[inline]
pub fn adler32(data: &[u8]) -> u32 {
    let mut h = simd_adler32::Adler32::new();
    h.write(data);
    h.finish()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_adler32_empty() {
        assert_eq!(adler32(&[]), 1);
    }

    #[test]
    fn test_adler32_known() {
        assert_eq!(adler32(b"Wikipedia"), 0x11E6_0398);
    }
}
