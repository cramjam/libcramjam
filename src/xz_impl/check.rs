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

/// Slice-by-8 tables for CRC32 — same idea as CRC64 above.
static CRC32_SLICE_TABLES: OnceLock<[[u32; 256]; 8]> = OnceLock::new();

fn crc32_slice_tables() -> &'static [[u32; 256]; 8] {
    CRC32_SLICE_TABLES.get_or_init(|| {
        let mut t = [[0u32; 256]; 8];
        for (i, slot) in t[0].iter_mut().enumerate() {
            let mut c = i as u32;
            for _ in 0..8 {
                c = if c & 1 != 0 { (c >> 1) ^ CRC32_POLY } else { c >> 1 };
            }
            *slot = c;
        }
        for k in 1..8 {
            for i in 0..256 {
                let prev = t[k - 1][i];
                t[k][i] = (prev >> 8) ^ t[0][(prev & 0xff) as usize];
            }
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
        let tables = crc32_slice_tables();
        let mut c = self.state;
        let mut chunks = data.chunks_exact(8);
        for chunk in &mut chunks {
            let word = u64::from_le_bytes(chunk.try_into().unwrap());
            let v_low = (c as u64) ^ (word & 0xFFFFFFFF);
            let v_high = word >> 32;
            c = tables[7][(v_low & 0xff) as usize]
                ^ tables[6][((v_low >> 8) & 0xff) as usize]
                ^ tables[5][((v_low >> 16) & 0xff) as usize]
                ^ tables[4][((v_low >> 24) & 0xff) as usize]
                ^ tables[3][(v_high & 0xff) as usize]
                ^ tables[2][((v_high >> 8) & 0xff) as usize]
                ^ tables[1][((v_high >> 16) & 0xff) as usize]
                ^ tables[0][((v_high >> 24) & 0xff) as usize];
        }
        let tail = chunks.remainder();
        for &b in tail {
            let idx = ((c ^ b as u32) & 0xff) as usize;
            c = (c >> 8) ^ tables[0][idx];
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

/// Slice-by-8 CRC64 — eight precomputed lookup tables, processes 8 input
/// bytes per iteration of the inner loop with 8 parallel table reads that
/// the CPU can pipeline.  Single-table fallback is used for the trailing
/// bytes that don't fill a full 8-byte block.
static CRC64_SLICE_TABLES: OnceLock<[[u64; 256]; 8]> = OnceLock::new();

fn crc64_slice_tables() -> &'static [[u64; 256]; 8] {
    CRC64_SLICE_TABLES.get_or_init(|| {
        let mut t = [[0u64; 256]; 8];
        // Table[0] is the standard reflected CRC64 table.
        for (i, slot) in t[0].iter_mut().enumerate() {
            let mut c = i as u64;
            for _ in 0..8 {
                c = if c & 1 != 0 { (c >> 1) ^ CRC64_POLY } else { c >> 1 };
            }
            *slot = c;
        }
        // Each subsequent table[k] = (table[k-1] >> 8) ^ table[0][bottom byte].
        for k in 1..8 {
            for i in 0..256 {
                let prev = t[k - 1][i];
                t[k][i] = (prev >> 8) ^ t[0][(prev & 0xff) as usize];
            }
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
        let tables = crc64_slice_tables();
        let mut c = self.state;
        // Slice-by-8 main loop.
        let mut chunks = data.chunks_exact(8);
        for chunk in &mut chunks {
            // Read 8 bytes as a little-endian u64, XOR with the current
            // state, and look up each byte in the corresponding table.
            let word = u64::from_le_bytes(chunk.try_into().unwrap());
            let v = c ^ word;
            c = tables[7][(v & 0xff) as usize]
                ^ tables[6][((v >> 8) & 0xff) as usize]
                ^ tables[5][((v >> 16) & 0xff) as usize]
                ^ tables[4][((v >> 24) & 0xff) as usize]
                ^ tables[3][((v >> 32) & 0xff) as usize]
                ^ tables[2][((v >> 40) & 0xff) as usize]
                ^ tables[1][((v >> 48) & 0xff) as usize]
                ^ tables[0][((v >> 56) & 0xff) as usize];
        }
        // Trailing bytes via the slow per-byte path using table[0].
        let tail = chunks.remainder();
        for &b in tail {
            let idx = ((c ^ b as u64) & 0xff) as usize;
            c = (c >> 8) ^ tables[0][idx];
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
