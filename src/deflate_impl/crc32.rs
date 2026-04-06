//! CRC-32 as specified in RFC 1952 (ISO 3309, ITU-T V.42)
//!
//! Delegates to `crc32fast` which uses hardware CLMUL acceleration when
//! available, falling back to a fast software implementation.

/// Compute CRC-32 of a byte slice.
#[inline]
pub fn crc32(data: &[u8]) -> u32 {
    let mut h = crc32fast::Hasher::new();
    h.update(data);
    h.finalize()
}

/// Incremental CRC-32 computation.
pub struct Crc32(crc32fast::Hasher);

impl Crc32 {
    #[inline]
    pub fn new() -> Self {
        Self(crc32fast::Hasher::new())
    }

    #[inline]
    pub fn update(&mut self, data: &[u8]) {
        self.0.update(data);
    }

    #[inline]
    pub fn finalize(self) -> u32 {
        self.0.finalize()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_crc32_empty() {
        assert_eq!(crc32(&[]), 0x0000_0000);
    }

    #[test]
    fn test_crc32_known() {
        assert_eq!(crc32(b"123456789"), 0xCBF4_3926);
    }

    #[test]
    fn test_crc32_incremental() {
        let data = b"Hello, world!";
        let expected = crc32(data);
        let mut c = Crc32::new();
        c.update(&data[..5]);
        c.update(&data[5..]);
        assert_eq!(c.finalize(), expected);
    }
}
