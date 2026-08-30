//! Integrity-check primitives used by the .xz container.
//!
//! XZ supports None / CRC32 / CRC64 / SHA-256. CRC32 is the standard
//! reflected CRC-32 (`0xEDB88320`, same as gzip) and CRC64 the reflected
//! CRC-64 with polynomial `0xC96C5795D7870F42` (CRC-64/XZ). Both delegate to
//! SIMD (CLMUL) implementations: a byte/slice table CRC64 costs ~3 Ir/byte,
//! which on incompressible input was most of the decode time.

/// CRC-32 (gzip / xz flavor).
#[inline]
pub fn crc32(data: &[u8]) -> u32 {
    let mut h = crc32fast::Hasher::new();
    h.update(data);
    h.finalize()
}

/// CRC-64/XZ.
#[inline]
pub fn crc64(data: &[u8]) -> u64 {
    crc_fast::checksum(crc_fast::CrcAlgorithm::Crc64Xz, data)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn crc32_known_vector() {
        assert_eq!(crc32(b"123456789"), 0xCBF43926);
        assert_eq!(crc32(b""), 0);
    }

    #[test]
    fn crc64_known_vector() {
        assert_eq!(crc64(b"123456789"), 0x995DC9BBDF1939FA);
        assert_eq!(crc64(b""), 0);
        // Long input exercises the folding loops.
        let data: Vec<u8> = (0..100_000u32).map(|i| (i.wrapping_mul(2654435761) >> 13) as u8).collect();
        let mut c: u64 = !0;
        for &b in &data {
            c ^= b as u64;
            for _ in 0..8 {
                c = if c & 1 != 0 { (c >> 1) ^ 0xC96C5795D7870F42 } else { c >> 1 };
            }
        }
        assert_eq!(crc64(&data), !c);
    }
}
