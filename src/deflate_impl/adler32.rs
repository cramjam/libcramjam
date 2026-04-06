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

/// Incremental Adler-32 computation.
pub struct Adler32(simd_adler32::Adler32);

impl Adler32 {
    #[inline]
    pub fn new() -> Self {
        Self(simd_adler32::Adler32::new())
    }

    #[inline]
    pub fn update(&mut self, data: &[u8]) {
        self.0.write(data);
    }

    #[inline]
    pub fn finalize(self) -> u32 {
        self.0.finish()
    }
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

    #[test]
    fn test_adler32_incremental() {
        let data = b"Hello, world!";
        let expected = adler32(data);
        let mut a = Adler32::new();
        a.update(&data[..5]);
        a.update(&data[5..]);
        assert_eq!(a.finalize(), expected);
    }
}
