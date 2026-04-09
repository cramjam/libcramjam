//! Integrity-check primitives used by the .xz container.
//!
//! XZ supports None / CRC32 / CRC64 / SHA-256.  CRC32 here is the standard
//! reflected CRC-32 with polynomial `0xEDB88320` (same as gzip), and CRC64
//! is the reflected CRC-64 with polynomial `0xC96C5795D7870F42` defined by
//! the .xz format spec.

use std::sync::OnceLock;

// =========================================================================
// CRC32 (gzip flavor — reflected, init 0xFFFFFFFF, finalize ^= 0xFFFFFFFF)
// =========================================================================

const CRC32_POLY: u32 = 0xEDB88320;

static CRC32_TABLE: OnceLock<[u32; 256]> = OnceLock::new();

fn crc32_table() -> &'static [u32; 256] {
    CRC32_TABLE.get_or_init(|| {
        let mut t = [0u32; 256];
        for (i, slot) in t.iter_mut().enumerate() {
            let mut c = i as u32;
            for _ in 0..8 {
                c = if c & 1 != 0 { (c >> 1) ^ CRC32_POLY } else { c >> 1 };
            }
            *slot = c;
        }
        t
    })
}

#[derive(Clone)]
pub struct Crc32 {
    state: u32,
}

impl Crc32 {
    pub fn new() -> Self {
        Self { state: 0xFFFFFFFF }
    }

    pub fn update(&mut self, data: &[u8]) {
        let tbl = crc32_table();
        let mut c = self.state;
        for &b in data {
            let idx = ((c ^ b as u32) & 0xff) as usize;
            c = (c >> 8) ^ tbl[idx];
        }
        self.state = c;
    }

    pub fn finalize(self) -> u32 {
        self.state ^ 0xFFFFFFFF
    }
}

impl Default for Crc32 {
    fn default() -> Self {
        Self::new()
    }
}

pub fn crc32(data: &[u8]) -> u32 {
    let mut c = Crc32::new();
    c.update(data);
    c.finalize()
}

// =========================================================================
// CRC64 (xz flavor — reflected, init 0xFFFFFFFFFFFFFFFF, final XOR same)
// =========================================================================

const CRC64_POLY: u64 = 0xC96C5795D7870F42;

static CRC64_TABLE: OnceLock<[u64; 256]> = OnceLock::new();

fn crc64_table() -> &'static [u64; 256] {
    CRC64_TABLE.get_or_init(|| {
        let mut t = [0u64; 256];
        for (i, slot) in t.iter_mut().enumerate() {
            let mut c = i as u64;
            for _ in 0..8 {
                c = if c & 1 != 0 { (c >> 1) ^ CRC64_POLY } else { c >> 1 };
            }
            *slot = c;
        }
        t
    })
}

#[derive(Clone)]
pub struct Crc64 {
    state: u64,
}

impl Crc64 {
    pub fn new() -> Self {
        Self { state: 0xFFFFFFFFFFFFFFFF }
    }

    pub fn update(&mut self, data: &[u8]) {
        let tbl = crc64_table();
        let mut c = self.state;
        for &b in data {
            let idx = ((c ^ b as u64) & 0xff) as usize;
            c = (c >> 8) ^ tbl[idx];
        }
        self.state = c;
    }

    pub fn finalize(self) -> u64 {
        self.state ^ 0xFFFFFFFFFFFFFFFF
    }
}

impl Default for Crc64 {
    fn default() -> Self {
        Self::new()
    }
}

pub fn crc64(data: &[u8]) -> u64 {
    let mut c = Crc64::new();
    c.update(data);
    c.finalize()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn crc32_known_vector() {
        // CRC-32 of "123456789" should be 0xCBF43926.
        assert_eq!(crc32(b"123456789"), 0xCBF43926);
    }

    #[test]
    fn crc32_empty() {
        assert_eq!(crc32(b""), 0);
    }

    #[test]
    fn crc64_known_vector() {
        // CRC-64-XZ of "123456789" is 0x995DC9BBDF1939FA per the xz spec.
        assert_eq!(crc64(b"123456789"), 0x995DC9BBDF1939FA);
    }

    #[test]
    fn crc64_empty() {
        assert_eq!(crc64(b""), 0);
    }
}
