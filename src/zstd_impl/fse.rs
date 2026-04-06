//! FSE (Finite State Entropy) decoding for Zstandard (RFC 8878 Section 4.1).
//!
//! FSE is a table-based ANS (Asymmetric Numeral Systems) codec. Each entry in
//! the decoding table contains: symbol, number of bits to read, and a baseline
//! for computing the next state.

use std::io;

use super::bits::{ForwardByteReader, ReverseBitReader};

/// Maximum accuracy log for FSE tables.
pub const FSE_MAX_ACCURACY_LOG: u32 = 9;

/// A single entry in an FSE decoding table.
#[derive(Clone, Copy, Default)]
pub struct FseEntry {
    pub symbol: u8,
    pub num_bits: u8,
    pub baseline: u16,
}

/// FSE decoding table.
pub struct FseTable {
    pub table: Vec<FseEntry>,
    pub accuracy_log: u32,
}

impl FseTable {
    /// Build a decoding table from a distribution of symbol weights.
    ///
    /// `norm_weights[symbol]` is the normalized probability for that symbol.
    /// A weight of -1 means "less than 1" (low-probability symbol).
    pub fn from_weights(weights: &[i16], accuracy_log: u32) -> io::Result<Self> {
        let table_size = 1usize << accuracy_log;
        let mut table = vec![FseEntry::default(); table_size];

        // Phase 1: Spread symbols across the table.
        // Symbols with weight == -1 get exactly 1 slot, placed at the end.
        // Other symbols get `weight` slots, distributed evenly.
        let mut high_threshold = table_size;
        for (sym, &w) in weights.iter().enumerate() {
            if w == -1 {
                high_threshold -= 1;
                table[high_threshold].symbol = sym as u8;
            }
        }

        let step = (table_size >> 1) + (table_size >> 3) + 3;
        let mask = table_size - 1;
        let mut pos = 0usize;

        for (sym, &w) in weights.iter().enumerate() {
            if w <= 0 {
                continue;
            }
            for _ in 0..w as usize {
                table[pos].symbol = sym as u8;
                // Advance position, skipping slots already used by high_threshold symbols.
                loop {
                    pos = (pos + step) & mask;
                    if pos < high_threshold {
                        break;
                    }
                }
            }
        }

        // Phase 2: Compute num_bits and baseline for each entry.
        let mut symbol_next = vec![0u16; weights.len()];
        for (sym, &w) in weights.iter().enumerate() {
            let s = if w == -1 { 1 } else { w as u16 };
            symbol_next[sym] = s;
        }

        for i in 0..table_size {
            let sym = table[i].symbol as usize;
            let s = symbol_next[sym];
            let w = if weights[sym] == -1 { 1i16 } else { weights[sym] };
            let nb = (accuracy_log - highest_bit(s as u32)) as u8;
            table[i].num_bits = nb;
            table[i].baseline = ((s as u32) << nb) as u16 - table_size as u16;
            symbol_next[sym] = s + 1;
        }

        Ok(Self {
            table,
            accuracy_log,
        })
    }

    /// Decode the FSE table description from a compressed bitstream.
    ///
    /// Returns the table and the number of bytes consumed.
    pub fn decode_table(reader: &mut ForwardByteReader, max_symbol: u32, max_accuracy_log: u32) -> io::Result<Self> {
        let data = &reader.data[reader.pos..];
        if data.is_empty() {
            return Err(io::Error::new(io::ErrorKind::InvalidData, "zstd: empty FSE table"));
        }

        let mut bit_pos = 0usize;

        let read_bits_fwd = |data: &[u8], bit_pos: &mut usize, n: u32| -> u32 {
            let byte_idx = *bit_pos / 8;
            let bit_idx = *bit_pos % 8;
            if byte_idx + 4 <= data.len() {
                let val = u32::from_le_bytes(data[byte_idx..byte_idx + 4].try_into().unwrap());
                let result = (val >> bit_idx) & ((1u32 << n) - 1);
                *bit_pos += n as usize;
                result
            } else {
                // Slow path for end of data.
                let mut val = 0u64;
                for i in byte_idx..data.len().min(byte_idx + 8) {
                    val |= (data[i] as u64) << ((i - byte_idx) * 8);
                }
                let result = ((val >> bit_idx) & ((1u64 << n) - 1)) as u32;
                *bit_pos += n as usize;
                result
            }
        };

        let accuracy_log = read_bits_fwd(data, &mut bit_pos, 4) + 5;
        if accuracy_log > max_accuracy_log {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "zstd: FSE accuracy log exceeds maximum",
            ));
        }
        let table_size = 1u32 << accuracy_log;
        let mut remaining = table_size as i32 + 1;
        let mut weights = Vec::new();
        let max_sym = max_symbol as usize + 1;

        while remaining > 1 && weights.len() < max_sym {
            let threshold = highest_bit(remaining as u32) + 1;
            let mut bits = threshold - 1;
            let lower_mask = (1i32 << bits) - 1;
            let upper_mask = (1i32 << threshold) - 1;

            let small = read_bits_fwd(data, &mut bit_pos, bits) as i32;

            // Two-range encoding: small values are `bits` bits, larger values are `threshold` bits.
            let boundary = upper_mask - remaining;
            let value = if small < boundary {
                small
            } else {
                let extra = read_bits_fwd(data, &mut bit_pos, 1) as i32;
                let big = (small << 1) + extra;
                if big < upper_mask {
                    big - boundary
                } else {
                    big - upper_mask
                }
            };

            let prob = value - 1; // prob can be -1 (for "less than 1")
            weights.push(prob as i16);

            if prob == 0 {
                // Repeat zeros.
                loop {
                    let repeat = read_bits_fwd(data, &mut bit_pos, 2) as usize;
                    for _ in 0..repeat {
                        weights.push(0);
                    }
                    if repeat < 3 {
                        break;
                    }
                }
            }

            remaining -= if prob < 0 { 1 } else { prob as i32 };
        }

        // Fill remaining symbols with 0.
        while weights.len() < max_sym {
            weights.push(0);
        }

        let bytes_consumed = (bit_pos + 7) / 8;
        reader.pos += bytes_consumed;

        Self::from_weights(&weights, accuracy_log)
    }

    /// Peek at the symbol for the current state.
    #[inline(always)]
    pub fn symbol(&self, state: u32) -> u8 {
        self.table[state as usize].symbol
    }

    /// Advance to the next state by reading bits from the backward bitstream.
    #[inline(always)]
    pub fn next_state(&self, state: u32, bits: &mut ReverseBitReader) -> u32 {
        let entry = &self.table[state as usize];
        let low_bits = bits.read_bits(entry.num_bits as u32);
        entry.baseline as u32 + low_bits
    }
}

/// Build an FSE table from a single repeated symbol (RLE mode).
pub fn fse_rle_table(symbol: u8) -> FseTable {
    FseTable {
        table: vec![FseEntry {
            symbol,
            num_bits: 0,
            baseline: 0,
        }],
        accuracy_log: 0,
    }
}

/// Highest set bit position (0-indexed). `highest_bit(1) = 0, highest_bit(4) = 2`.
#[inline(always)]
fn highest_bit(v: u32) -> u32 {
    debug_assert!(v > 0);
    31 - v.leading_zeros()
}

// ---------------------------------------------------------------------------
// Predefined FSE tables (RFC 8878 Appendix A)
// ---------------------------------------------------------------------------

/// Build the predefined Literals Length FSE table (accuracy_log = 6).
pub fn predefined_litlen_table() -> FseTable {
    // From RFC 8878 Appendix A, Table 14.
    static WEIGHTS: [i16; 36] = [
        4, 3, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 1, 1, 1,
        2, 2, 2, 2, 2, 2, 2, 2, 2, 3, 2, 1, 1, 1, 1, 1,
        -1, -1, -1, -1,
    ];
    FseTable::from_weights(&WEIGHTS, 6).unwrap()
}

/// Build the predefined Match Length FSE table (accuracy_log = 6).
pub fn predefined_matchlen_table() -> FseTable {
    // From RFC 8878 Appendix A, Table 16.
    static WEIGHTS: [i16; 53] = [
        1, 4, 3, 2, 2, 2, 2, 2, 2, 1, 1, 1, 1, 1, 1, 1,
        1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
        1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, -1, -1,
        -1, -1, -1, -1, -1,
    ];
    FseTable::from_weights(&WEIGHTS, 6).unwrap()
}

/// Build the predefined Offset FSE table (accuracy_log = 5).
pub fn predefined_offset_table() -> FseTable {
    // From RFC 8878 Appendix A, Table 18.
    static WEIGHTS: [i16; 29] = [
        1, 1, 1, 1, 1, 1, 2, 2, 2, 1, 1, 1, 1, 1, 1, 1,
        1, 1, 1, 1, 1, 1, 1, 1, -1, -1, -1, -1, -1,
    ];
    FseTable::from_weights(&WEIGHTS, 5).unwrap()
}

// ---------------------------------------------------------------------------
// Value decoding tables for literals length, match length, and offset
// (RFC 8878 Section 3.1.2.1.2)
// ---------------------------------------------------------------------------

/// (baseline, extra_bits) for Literals_Length codes 0..35.
pub const LITLEN_TABLE: [(u32, u8); 36] = [
    (0, 0), (1, 0), (2, 0), (3, 0), (4, 0), (5, 0), (6, 0), (7, 0),
    (8, 0), (9, 0), (10, 0), (11, 0), (12, 0), (13, 0), (14, 0), (15, 0),
    (16, 1), (18, 1), (20, 1), (22, 1), (24, 2), (28, 2), (32, 3), (40, 3),
    (48, 4), (64, 6), (128, 7), (256, 8), (512, 9), (1024, 10), (2048, 11), (4096, 12),
    (8192, 13), (16384, 14), (32768, 15), (65536, 16),
];

/// (baseline, extra_bits) for Match_Length codes 0..52.
pub const MATCHLEN_TABLE: [(u32, u8); 53] = [
    (3, 0), (4, 0), (5, 0), (6, 0), (7, 0), (8, 0), (9, 0), (10, 0),
    (11, 0), (12, 0), (13, 0), (14, 0), (15, 0), (16, 0), (17, 0), (18, 0),
    (19, 0), (20, 0), (21, 0), (22, 0), (23, 0), (24, 0), (25, 0), (26, 0),
    (27, 0), (28, 0), (29, 0), (30, 0), (31, 0), (32, 0), (33, 0), (34, 0),
    (35, 1), (37, 1), (39, 1), (41, 1), (43, 2), (47, 2), (51, 3), (59, 3),
    (67, 4), (83, 4), (99, 5), (131, 7), (259, 8), (515, 9), (1027, 10), (2051, 11),
    (4099, 12), (8195, 13), (16387, 14), (32771, 15), (65539, 16),
];
