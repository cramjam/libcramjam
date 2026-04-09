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
}
