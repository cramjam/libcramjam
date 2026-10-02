//! Huffman coding for DEFLATE (RFC 1951 section 3.2.2)
//!
//! Provides both decoding (for inflate) and encoding (for deflate compress).

use std::io;

use super::bitreader::BitReader;

// ---------------------------------------------------------------------------
// Shared utility
// ---------------------------------------------------------------------------

/// Reverse the lowest `num_bits` bits of `value`.
#[inline(always)]
pub fn reverse_bits(value: u32, num_bits: u32) -> u32 {
    value.reverse_bits() >> (32 - num_bits)
}

/// Compute canonical Huffman codes from code lengths (RFC 1951 section 3.2.2).
///
/// Returns `(reversed_code, code_length)` for every symbol.  The reversed code
/// is ready for direct use with [`BitWriter::write_bits`] (LSB-first packing)
/// and as a lookup-table index for decoding.
pub fn canonical_codes(lengths: &[u8]) -> Vec<(u32, u8)> {
    let n = lengths.len();
    let max_bits = match lengths.iter().copied().max() {
        Some(m) if m > 0 => m,
        _ => return vec![(0, 0); n],
    };

    // 1. Count codes of each length
    let mut bl_count = vec![0u32; max_bits as usize + 1];
    for &len in lengths {
        if len > 0 {
            bl_count[len as usize] += 1;
        }
    }

    // 2. Starting code for each bit length
    let mut next_code = vec![0u32; max_bits as usize + 1];
    let mut code = 0u32;
    for bits in 1..=max_bits as usize {
        code = (code + bl_count[bits - 1]) << 1;
        next_code[bits] = code;
    }

    // 3. Assign reversed codes
    let mut codes = vec![(0u32, 0u8); n];
    for sym in 0..n {
        let len = lengths[sym];
        if len == 0 {
            continue;
        }
        let c = next_code[len as usize];
        next_code[len as usize] += 1;
        codes[sym] = (reverse_bits(c, len as u32), len);
    }
    codes
}

// ---------------------------------------------------------------------------
// Decoder (two-level table, libdeflate-style packed entries)
// ---------------------------------------------------------------------------
//
// Every table entry is a `u32`:
//
//   bits  0..=5   code length to consume (full length; for subtable
//                 pointers: the main-table bits)
//   bits  8..=13  extra bits (length/distance codes) or subtable bits
//                 (pointer entries)
//   bits 14..=15  kind: 0 literal, 1 length/distance base, 2 special
//                 (value 256 = end-of-block, anything else = invalid code),
//                 3 subtable pointer
//   bits 16..=31  literal byte / length or distance base / subtable start
//
// Codes no longer than `main_bits` are resolved with one lookup; longer
// codes hit a second, exactly-sized subtable appended after the main table.

pub const KIND_LITERAL: u32 = 0;
pub const KIND_BASE: u32 = 1;
pub const KIND_SPECIAL: u32 = 2;
pub const KIND_SUB: u32 = 3;
pub const KIND_SHIFT: u32 = 14;
pub const KIND_MASK: u32 = 3 << KIND_SHIFT;
pub const LEN_MASK: u32 = 0x3F;
pub const EXTRA_SHIFT: u32 = 8;
pub const EXTRA_MASK: u32 = 0x3F;
pub const EOB_ENTRY_VALUE: u32 = 256;

#[inline(always)]
const fn entry(kind: u32, value: u32, extra: u32, len: u32) -> u32 {
    (value << 16) | (kind << KIND_SHIFT) | (extra << EXTRA_SHIFT) | len
}

/// Marker for bit patterns that don't correspond to any code.
const INVALID: u32 = entry(KIND_SPECIAL, 0xFFFF, 0, 0);

/// Which alphabet a table decodes — determines how symbols map to entries.
#[derive(Clone, Copy, PartialEq, Eq)]
pub enum Alphabet {
    /// Literal/length: 0..=255 literals, 256 EOB, 257..=285 lengths.
    LitLen,
    /// Distance codes 0..=29.
    Distance,
    /// Code-length codes 0..=18 (header precode).
    Precode,
}

fn symbol_entry(alphabet: Alphabet, sym: usize, len: u32) -> u32 {
    use super::tables;
    match alphabet {
        Alphabet::LitLen => {
            if sym < 256 {
                entry(KIND_LITERAL, sym as u32, 0, len)
            } else if sym == 256 {
                entry(KIND_SPECIAL, EOB_ENTRY_VALUE, 0, len)
            } else if sym <= 285 {
                let i = sym - 257;
                entry(KIND_BASE, tables::LENGTH_BASE[i] as u32, tables::LENGTH_EXTRA[i] as u32, len)
            } else {
                entry(KIND_SPECIAL, 0xFFFF, 0, len)
            }
        }
        Alphabet::Distance => {
            if sym < 30 {
                entry(KIND_BASE, tables::DISTANCE_BASE[sym] as u32, tables::DISTANCE_EXTRA[sym] as u32, len)
            } else {
                entry(KIND_SPECIAL, 0xFFFF, 0, len)
            }
        }
        Alphabet::Precode => entry(KIND_LITERAL, sym as u32, 0, len),
    }
}

/// Fast two-level table Huffman decoder.
pub struct HuffmanDecoder {
    table: Vec<u32>,
    pub main_bits: u32,
}

impl HuffmanDecoder {
    /// Build a decoder from code lengths.  `lengths[i]` is the code length for
    /// symbol `i`; zero means the symbol does not occur.  `main_bits` is the
    /// first-level table width (10 for lit/len, 8 for distance, 7 for the
    /// precode).  Over-subscribed codes are rejected; incomplete codes are
    /// allowed (unused bit patterns decode as invalid).
    pub fn from_lengths(lengths: &[u8], alphabet: Alphabet, main_bits: u32) -> io::Result<Self> {
        debug_assert!(main_bits <= 10);
        let mut kraft = 0u32;
        for &len in lengths {
            if len > 15 {
                return Err(io::Error::new(io::ErrorKind::InvalidData, "huffman code length exceeds 15"));
            }
            if len > 0 {
                kraft += 1 << (15 - len);
            }
        }
        if kraft > 1 << 15 {
            return Err(io::Error::new(io::ErrorKind::InvalidData, "over-subscribed huffman code"));
        }

        let main_size = 1usize << main_bits;
        let main_mask = main_size - 1;
        let mut table = vec![INVALID; main_size];
        let codes = canonical_codes(lengths);

        // Pass 1: short codes go straight into the main table; record the
        // longest code behind each main-table prefix for the long ones.
        let mut group_max = [0u8; 1024];
        for (sym, &(rev, len)) in codes.iter().enumerate() {
            if len == 0 {
                continue;
            }
            let len = len as u32;
            if len <= main_bits {
                let e = symbol_entry(alphabet, sym, len);
                let step = 1usize << len;
                let mut idx = rev as usize;
                while idx < main_size {
                    table[idx] = e;
                    idx += step;
                }
            } else {
                let p = rev as usize & main_mask;
                group_max[p] = group_max[p].max(len as u8);
            }
        }

        // Pass 2: allocate one subtable per used prefix.
        let mut sub_start = [0u32; 1024];
        for p in 0..main_size {
            if group_max[p] > 0 {
                let sub_bits = group_max[p] as u32 - main_bits;
                let start = table.len();
                table.resize(start + (1 << sub_bits), INVALID);
                table[p] = entry(KIND_SUB, start as u32, sub_bits, main_bits);
                sub_start[p] = start as u32;
            }
        }

        // Pass 3: fill the subtables.
        for (sym, &(rev, len)) in codes.iter().enumerate() {
            let len = len as u32;
            if len <= main_bits {
                continue;
            }
            let p = rev as usize & main_mask;
            let start = sub_start[p] as usize;
            let sub_bits = group_max[p] as u32 - main_bits;
            let sub_len = len - main_bits;
            let e = symbol_entry(alphabet, sym, len);
            let step = 1usize << sub_len;
            let mut idx = rev as usize >> main_bits;
            while idx < 1usize << sub_bits {
                table[start + idx] = e;
                idx += step;
            }
        }

        Ok(Self { table, main_bits })
    }

    /// Resolve the entry for the bits currently in `reader` (which must hold
    /// at least 15 valid bits) *without* consuming anything.  The returned
    /// entry's low 6 bits are the code length to consume.
    ///
    /// `MB` must equal the `main_bits` the table was built with (a const so
    /// the masks fold into immediates in the hot loop).
    #[inline(always)]
    pub fn lookup<const MB: u32>(&self, reader: &BitReader) -> u32 {
        debug_assert_eq!(MB, self.main_bits);
        // The main table has `1 << MB` entries and `peek` masks to that
        // width; a subtable's start + (sub_bits-wide index) lies inside the
        // exactly-sized subtable appended in `from_lengths`. The bounds
        // checks here measured as free (inflate is table-load bound).
        let mut e = self.table[reader.peek(MB) as usize];
        if e & KIND_MASK == KIND_SUB << KIND_SHIFT {
            let sub_bits = (e >> EXTRA_SHIFT) & EXTRA_MASK;
            let idx = (reader.peek(MB + sub_bits) >> MB) as usize;
            e = self.table[(e >> 16) as usize + idx];
        }
        e
    }

    /// Decode one symbol (checked; used for headers).  Returns the raw
    /// symbol index for literal-kind entries.
    pub fn decode(&self, reader: &mut BitReader) -> io::Result<u16> {
        if !reader.refill() {
            return Err(io::Error::new(io::ErrorKind::UnexpectedEof, "unexpected end of deflate stream"));
        }
        let e = match self.main_bits {
            7 => self.lookup::<7>(reader),
            8 => self.lookup::<8>(reader),
            _ => self.lookup::<10>(reader),
        };
        if e & KIND_MASK != KIND_LITERAL << KIND_SHIFT {
            return Err(io::Error::new(io::ErrorKind::InvalidData, "invalid huffman code"));
        }
        reader.consume(e & LEN_MASK);
        if reader.overread() {
            return Err(io::Error::new(io::ErrorKind::UnexpectedEof, "unexpected end of deflate stream"));
        }
        Ok((e >> 16) as u16)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_reverse_bits() {
        assert_eq!(reverse_bits(0b110, 3), 0b011);
        assert_eq!(reverse_bits(0b1010, 4), 0b0101);
        assert_eq!(reverse_bits(0b1, 1), 0b1);
    }

    #[test]
    fn test_canonical_codes_fixed() {
        // Verify a few entries of the fixed literal/length codes.
        let lengths = super::super::tables::fixed_literal_lengths();
        let codes = canonical_codes(&lengths);
        // Symbol 0 has length 8 – its canonical code (non-reversed) is 00110000 = 48.
        // Reversed: reverse_bits(48, 8) = ?
        // 48 = 0b00110000, reversed 8 bits = 0b00001100 = 12
        assert_eq!(codes[0], (reverse_bits(48, 8), 8));
    }


}
