//! bzip2's CRC-32.
//!
//! Same polynomial as gzip's CRC-32 (0x04C11DB7) but **bytes are processed
//! MSB-first** instead of LSB-first.  That means we cannot reuse `crc32fast`
//! — that crate uses the bit-reversed convention.
//!
//! The implementation is the standard table-driven Sarwate algorithm with a
//! 256-entry lookup table built at first use.

use std::sync::OnceLock;

const POLY: u32 = 0x04C11DB7;

static TABLE: OnceLock<[u32; 256]> = OnceLock::new();

fn table() -> &'static [u32; 256] {
    TABLE.get_or_init(|| {
        let mut table = [0u32; 256];
        for (i, slot) in table.iter_mut().enumerate() {
            let mut c = (i as u32) << 24;
            for _ in 0..8 {
                if c & 0x8000_0000 != 0 {
                    c = (c << 1) ^ POLY;
                } else {
                    c <<= 1;
                }
            }
            *slot = c;
        }
        table
    })
}

/// Incremental CRC-32 (bzip2 flavor) — initial state is `0xFFFFFFFF`,
/// final value is XORed with `0xFFFFFFFF`.
pub struct Crc32 {
    state: u32,
}

impl Crc32 {
    pub fn new() -> Self {
        Self { state: 0xFFFFFFFF }
    }

    pub fn update(&mut self, data: &[u8]) {
        let tbl = table();
        let mut c = self.state;
        for &b in data {
            let idx = (((c >> 24) as u8) ^ b) as usize;
            c = (c << 8) ^ tbl[idx];
        }
        self.state = c;
    }

    /// Add a single byte to the CRC.  Slightly faster than calling
    /// `update(&[b])` from a tight loop.
    #[inline]
    pub fn update_byte(&mut self, b: u8) {
        let tbl = table();
        let idx = (((self.state >> 24) as u8) ^ b) as usize;
        self.state = (self.state << 8) ^ tbl[idx];
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

/// One-shot helper.
pub fn crc32_of(data: &[u8]) -> u32 {
    let mut c = Crc32::new();
    c.update(data);
    c.finalize()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Known-good vector from the bzip2 test corpus: CRC of "abc" = 0x648CBB73.
    #[test]
    fn known_vector_abc() {
        assert_eq!(crc32_of(b"abc"), 0x648C_BB73);
    }

    /// Empty input → 0.
    #[test]
    fn empty() {
        assert_eq!(crc32_of(b""), 0);
    }

    /// Cross-validation: same hash twice gives same value (sanity).
    #[test]
    fn deterministic() {
        let a = crc32_of(b"hello world");
        let b = crc32_of(b"hello world");
        assert_eq!(a, b);
    }

    /// Incremental update equals one-shot.
    #[test]
    fn incremental_matches_oneshot() {
        let data = b"The quick brown fox jumps over the lazy dog";
        let mut c = Crc32::new();
        c.update(&data[..10]);
        c.update(&data[10..20]);
        c.update(&data[20..]);
        assert_eq!(c.finalize(), crc32_of(data));
    }
}
