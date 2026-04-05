//! Huffman coding for DEFLATE (RFC 1951 section 3.2.2)
//!
//! Provides both decoding (for inflate) and encoding (for deflate compress).

use std::io;

use super::bitreader::BitReader;
use super::bitwriter::BitWriter;

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
// Decoder (table-based)
// ---------------------------------------------------------------------------

/// Fast table-based Huffman decoder.
///
/// The lookup table has `2^max_bits` entries.  Short codes are replicated so
/// that indexing with the next `max_bits` bits from the stream always yields
/// the correct symbol and code length.
pub struct HuffmanDecoder {
    table: Vec<(u16, u8)>, // (symbol, code_length)
    max_bits: u32,
}

impl HuffmanDecoder {
    /// Build a decoder from code lengths.  `lengths[i]` is the code length for
    /// symbol `i`; zero means the symbol does not occur.
    pub fn from_lengths(lengths: &[u8]) -> io::Result<Self> {
        let max_bits = lengths.iter().copied().max().unwrap_or(0) as u32;
        if max_bits == 0 {
            // Empty alphabet – decoder should never be called.
            return Ok(Self {
                table: Vec::new(),
                max_bits: 0,
            });
        }
        if max_bits > 15 {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "huffman code length exceeds 15",
            ));
        }

        let table_size = 1usize << max_bits;
        let mut table = vec![(0u16, 0u8); table_size];

        let codes = canonical_codes(lengths);
        for (sym, &(rev_code, len)) in codes.iter().enumerate() {
            if len == 0 {
                continue;
            }
            // Fill all entries whose lower `len` bits equal `rev_code`.
            let fill = 1usize << (max_bits - len as u32);
            for j in 0..fill {
                let idx = rev_code as usize | (j << len as usize);
                table[idx] = (sym as u16, len);
            }
        }

        Ok(Self { table, max_bits })
    }

    /// Decode one symbol from the bit reader.
    #[inline(always)]
    pub fn decode(&self, reader: &mut BitReader) -> io::Result<u16> {
        let bits = reader.peek_bits(self.max_bits)?;
        let (sym, len) = self.table[bits as usize];
        if len == 0 {
            return Self::decode_error(self.max_bits);
        }
        reader.consume(len as u32);
        Ok(sym)
    }

    #[cold]
    #[inline(never)]
    fn decode_error(max_bits: u32) -> io::Result<u16> {
        if max_bits == 0 {
            Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "decode from empty huffman table",
            ))
        } else {
            Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "invalid huffman code",
            ))
        }
    }
}

// ---------------------------------------------------------------------------
// Encoder helper
// ---------------------------------------------------------------------------

/// Encode a symbol using pre-computed reversed canonical codes.
#[inline]
pub fn encode_symbol(writer: &mut BitWriter, codes: &[(u32, u8)], sym: u16) {
    let (rev_code, len) = codes[sym as usize];
    debug_assert!(len > 0, "encoding symbol with zero-length code");
    writer.write_bits(rev_code, len as u32);
}

// ---------------------------------------------------------------------------
// Tree construction for compression
// ---------------------------------------------------------------------------

/// Build optimal Huffman code lengths from symbol frequencies, limited to
/// `max_len` bits.  Returns one length per symbol (0 if unused).
pub fn build_lengths(freqs: &[u32], max_len: u8) -> Vec<u8> {
    let n = freqs.len();
    let mut lengths = vec![0u8; n];

    let active: Vec<usize> = (0..n).filter(|&i| freqs[i] > 0).collect();

    match active.len() {
        0 => return lengths,
        1 => {
            lengths[active[0]] = 1;
            return lengths;
        }
        2 => {
            lengths[active[0]] = 1;
            lengths[active[1]] = 1;
            return lengths;
        }
        _ => {}
    }

    // Build Huffman tree via a min-heap.
    use std::cmp::Reverse;
    use std::collections::BinaryHeap;

    // Node ids: 0..n are leaf (symbol) nodes, n.. are internal nodes.
    let mut heap: BinaryHeap<Reverse<(u64, u32)>> = BinaryHeap::new();
    let mut children: Vec<(u32, u32)> = Vec::with_capacity(active.len());
    let mut next_id = n as u32;

    for &sym in &active {
        heap.push(Reverse((freqs[sym] as u64, sym as u32)));
    }

    while heap.len() > 1 {
        let Reverse((f1, id1)) = heap.pop().unwrap();
        let Reverse((f2, id2)) = heap.pop().unwrap();
        children.push((id1, id2));
        heap.push(Reverse((f1 + f2, next_id)));
        next_id += 1;
    }

    // Walk tree to compute depths.
    let root = heap.pop().unwrap().0 .1;
    let mut stack = vec![(root, 0u8)];
    while let Some((id, depth)) = stack.pop() {
        if (id as usize) < n {
            lengths[id as usize] = depth;
        } else {
            let (left, right) = children[(id as usize) - n];
            stack.push((left, depth + 1));
            stack.push((right, depth + 1));
        }
    }

    // Enforce depth limit.
    if *lengths.iter().max().unwrap_or(&0) > max_len {
        limit_lengths(&mut lengths, freqs, max_len);
    }

    lengths
}

/// Adjust code lengths so none exceeds `max_len` while keeping the code valid.
fn limit_lengths(lengths: &mut [u8], freqs: &[u32], max_len: u8) {
    // Sort symbols by frequency descending: most-frequent symbols keep short
    // codes; least-frequent are lengthened first.
    let mut syms: Vec<(u32, usize)> = lengths
        .iter()
        .enumerate()
        .filter(|(_, &l)| l > 0)
        .map(|(i, &_l)| (freqs[i], i))
        .collect();
    syms.sort_by(|a, b| b.0.cmp(&a.0));

    // Cap every length.
    for &(_, sym) in &syms {
        if lengths[sym] > max_len {
            lengths[sym] = max_len;
        }
    }

    // Kraft sum in units of 1 (with 2^max_len total).
    let target = 1u64 << max_len;
    let kraft = |lengths: &[u8], syms: &[(u32, usize)]| -> u64 {
        syms.iter()
            .map(|&(_, s)| 1u64 << (max_len - lengths[s]))
            .sum()
    };

    let mut current = kraft(lengths, &syms);

    // Lengthen the least-frequent symbols until within budget.
    let mut i = syms.len();
    while current > target && i > 0 {
        i -= 1;
        let sym = syms[i].1;
        while lengths[sym] < max_len && current > target {
            let old_cost = 1u64 << (max_len - lengths[sym]);
            lengths[sym] += 1;
            let new_cost = 1u64 << (max_len - lengths[sym]);
            current -= old_cost - new_cost;
        }
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

    #[test]
    fn test_build_lengths_simple() {
        // Two symbols, equal frequency
        let freqs = [10, 10];
        let lens = build_lengths(&freqs, 15);
        assert_eq!(lens, [1, 1]);
    }

    #[test]
    fn test_build_lengths_skewed() {
        let freqs = [100, 1, 1, 1];
        let lens = build_lengths(&freqs, 15);
        assert!(lens[0] < lens[1]);
    }
}
