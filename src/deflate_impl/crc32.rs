//! CRC-32 as specified in RFC 1952 (ISO 3309, ITU-T V.42)
//! Polynomial: 0xEDB88320 (reversed representation)

const CRC32_TABLE: [u32; 256] = {
    let mut table = [0u32; 256];
    let mut n = 0usize;
    while n < 256 {
        let mut c = n as u32;
        let mut k = 0;
        while k < 8 {
            if c & 1 != 0 {
                c = 0xEDB88320 ^ (c >> 1);
            } else {
                c >>= 1;
            }
            k += 1;
        }
        table[n] = c;
        n += 1;
    }
    table
};

/// Compute CRC-32 of a byte slice.
pub fn crc32(data: &[u8]) -> u32 {
    let mut crc = 0xFFFF_FFFFu32;
    for &byte in data {
        crc = CRC32_TABLE[((crc ^ byte as u32) & 0xFF) as usize] ^ (crc >> 8);
    }
    crc ^ 0xFFFF_FFFF
}

/// Incremental CRC-32 computation.
pub struct Crc32(u32);

impl Crc32 {
    pub fn new() -> Self {
        Self(0xFFFF_FFFF)
    }

    pub fn update(&mut self, data: &[u8]) {
        for &byte in data {
            self.0 = CRC32_TABLE[((self.0 ^ byte as u32) & 0xFF) as usize] ^ (self.0 >> 8);
        }
    }

    pub fn finalize(self) -> u32 {
        self.0 ^ 0xFFFF_FFFF
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
        // "123456789" has well-known CRC-32 = 0xCBF43926
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
