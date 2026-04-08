//! Huffman coding for Zstandard (RFC 8878 Section 4.2).
//!
//! Zstd Huffman uses weight-based tree description (not code lengths like DEFLATE)
//! and supports 1-stream or 4-stream parallel decoding for literals.

use std::io;

use super::bits::ReverseBitReader;
use super::fse;

/// Maximum number of Huffman symbols.
const HUF_MAX_SYMBOLS: usize = 256;
/// Maximum Huffman table log (number of bits for table lookup).
const HUF_MAX_TABLE_LOG: u32 = 12;

/// Huffman decoding table entry.
#[derive(Clone, Copy, Default)]
struct HufEntry {
    symbol: u8,
    num_bits: u8,
}

/// Huffman decoder.
pub struct HufTable {
    table: Vec<HufEntry>,
    max_bits: u32,
}

impl HufTable {
    /// Build from weights (as decoded from the header).
    /// Weight 0 means symbol not present. Weight w means code length = max_bits + 1 - w.
    /// The implicit last weight (for symbol == weights.len()) is inferred from the
    /// constraint that the sum of (1 << (w-1)) over all weights equals 2^max_bits.
    pub fn from_weights(weights: &[u8]) -> io::Result<Self> {
        if weights.is_empty() || weights.len() > 255 {
            return Err(io::Error::new(io::ErrorKind::InvalidData, "zstd: invalid Huffman weight count"));
        }
        for &w in weights {
            if w as u32 > HUF_MAX_TABLE_LOG {
                return Err(io::Error::new(io::ErrorKind::InvalidData, "zstd: Huffman weight exceeds limit"));
            }
        }

        // Sum of (1 << (w - 1)) for all explicitly-given symbols with w > 0.
        let weight_sum: u32 = weights
            .iter()
            .filter(|&&w| w > 0)
            .map(|&w| 1u32 << (w as u32 - 1))
            .sum();
        if weight_sum == 0 {
            return Err(io::Error::new(io::ErrorKind::InvalidData, "zstd: all Huffman weights are zero"));
        }

        // max_bits is the smallest integer such that 2^max_bits > weight_sum.
        // i.e. max_bits = highest_bit_set(weight_sum), where highest_bit_set is 1-indexed.
        let max_bits = highest_bit(weight_sum) + 1;
        if max_bits > HUF_MAX_TABLE_LOG {
            return Err(io::Error::new(io::ErrorKind::InvalidData, "zstd: Huffman table log too large"));
        }

        // Compute the implicit last weight from the leftover.
        let leftover = (1u32 << max_bits) - weight_sum;
        if !leftover.is_power_of_two() {
            return Err(io::Error::new(io::ErrorKind::InvalidData, "zstd: Huffman leftover is not a power of two"));
        }
        let last_weight = highest_bit(leftover) + 1; // 1-indexed log2 of leftover

        // Per-symbol code length (num_bits).  Symbols with weight 0 are unused.
        let total_symbols = weights.len() + 1;
        let mut num_bits_per_symbol = vec![0u8; total_symbols];
        for (sym, &w) in weights.iter().enumerate() {
            if w > 0 {
                num_bits_per_symbol[sym] = max_bits as u8 + 1 - w;
            }
        }
        num_bits_per_symbol[weights.len()] = max_bits as u8 + 1 - last_weight as u8;

        let table_size = 1usize << max_bits;
        let mut table = vec![HufEntry::default(); table_size];

        // Zstd canonical ordering (RFC 8878): longest codes occupy the LOW
        // end of the table, shortest codes occupy the HIGH end.  Within a
        // single code length, symbols are placed in ascending symbol order.
        //
        // Compute the starting table index for each code length using bit-rank
        // counts (matches the C reference and ruzstd).
        let mut bit_rank = vec![0u32; max_bits as usize + 1];
        for &b in &num_bits_per_symbol {
            if b > 0 {
                bit_rank[b as usize] += 1;
            }
        }
        let mut rank_start = vec![0usize; max_bits as usize + 1];
        // rank_start[max_bits] = 0 (longest codes first); going to shorter codes
        // we add the slots used by the previous (longer) ranks.
        for bits in (1..=max_bits as usize).rev() {
            let prev = if bits == max_bits as usize {
                0
            } else {
                rank_start[bits + 1] + bit_rank[bits + 1] as usize * (1 << (max_bits as usize - (bits + 1)))
            };
            rank_start[bits] = prev;
        }

        let mut rank_pos = rank_start.clone();
        for (sym, &b) in num_bits_per_symbol.iter().enumerate() {
            if b == 0 {
                continue;
            }
            let len = 1usize << (max_bits as u8 - b);
            let base = rank_pos[b as usize];
            for j in 0..len {
                table[base + j] = HufEntry { symbol: sym as u8, num_bits: b };
            }
            rank_pos[b as usize] += len;
        }

        Ok(Self { table, max_bits })
    }

    /// Decode Huffman tree description from the compressed stream.
    ///
    /// Returns the table and the number of bytes consumed from the input.
    pub fn decode_from_stream(data: &[u8]) -> io::Result<(Self, usize)> {
        if data.is_empty() {
            return Err(io::Error::new(io::ErrorKind::InvalidData, "zstd: empty Huffman header"));
        }

        let header_byte = data[0];
        if header_byte < 128 {
            // FSE-compressed weights.
            let compressed_size = header_byte as usize;
            if 1 + compressed_size > data.len() {
                return Err(io::Error::new(io::ErrorKind::InvalidData, "zstd: Huffman header truncated"));
            }
            let weights = decode_weights_fse(&data[1..1 + compressed_size])?;
            let table = Self::from_weights(&weights)?;
            Ok((table, 1 + compressed_size))
        } else {
            // Direct representation: weights are packed as 4-bit values.
            let num_symbols = (header_byte as usize) - 127;
            let num_bytes = (num_symbols + 1) / 2;
            if 1 + num_bytes > data.len() {
                return Err(io::Error::new(io::ErrorKind::InvalidData, "zstd: Huffman weights truncated"));
            }
            let mut weights = Vec::with_capacity(num_symbols);
            for i in 0..num_symbols {
                let byte = data[1 + i / 2];
                let w = if i % 2 == 0 { byte >> 4 } else { byte & 0x0F };
                weights.push(w);
            }
            let table = Self::from_weights(&weights)?;
            Ok((table, 1 + num_bytes))
        }
    }

    /// Decode a single symbol from a backward bitstream.
    #[inline(always)]
    pub fn decode_symbol(&self, bits: &mut ReverseBitReader) -> u8 {
        bits.ensure_bits(self.max_bits);
        let idx = bits.peek_bits(self.max_bits);
        let entry = self.table[idx as usize];
        bits.consume(entry.num_bits as u32);
        entry.symbol
    }
}

/// Decode Huffman weights using a 2-state FSE-compressed backward bitstream
/// (RFC 8878 Section 4.2.1.1).  Two interleaved decoder states share one
/// distribution table; symbols are emitted alternately until the bitstream
/// is exhausted.
fn decode_weights_fse(data: &[u8]) -> io::Result<Vec<u8>> {
    let mut reader = super::bits::ForwardByteReader::new(data);
    let fse_table = fse::FseTable::decode_table(&mut reader, 255, 6)?;
    let remaining = &data[reader.position()..];

    let mut bits = ReverseBitReader::new(remaining)?;
    bits.skip_padding_bits()?;

    let mut state1 = bits.get_bits(fse_table.accuracy_log);
    let mut state2 = bits.get_bits(fse_table.accuracy_log);

    let mut weights = Vec::with_capacity(64);
    // Loop: emit, then update.  When `update` would overrun, the OTHER state
    // still holds one final unread symbol.
    loop {
        weights.push(fse_table.symbol(state1));
        state1 = fse_table.next_state(state1, &mut bits);
        if bits.bits_remaining() < 0 {
            weights.push(fse_table.symbol(state2));
            break;
        }

        weights.push(fse_table.symbol(state2));
        state2 = fse_table.next_state(state2, &mut bits);
        if bits.bits_remaining() < 0 {
            weights.push(fse_table.symbol(state1));
            break;
        }

        if weights.len() > 255 {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "zstd: too many Huffman weights",
            ));
        }
    }
    Ok(weights)
}

/// Decode Huffman-compressed literals using 1 stream.
pub fn decode_literals_1stream(table: &HufTable, data: &[u8], regen_size: usize) -> io::Result<Vec<u8>> {
    let mut bits = ReverseBitReader::new(data)?;
    bits.skip_padding_bits()?;
    let mut output = Vec::with_capacity(regen_size);
    while output.len() < regen_size {
        output.push(table.decode_symbol(&mut bits));
    }
    Ok(output)
}

/// Decode Huffman-compressed literals using 4 streams.
pub fn decode_literals_4stream(table: &HufTable, data: &[u8], regen_size: usize) -> io::Result<Vec<u8>> {
    if data.len() < 6 {
        return Err(io::Error::new(io::ErrorKind::InvalidData, "zstd: 4-stream Huffman header too short"));
    }
    // Jump table: 3 u16 LE values giving the sizes of streams 1-3.
    let s1_size = u16::from_le_bytes([data[0], data[1]]) as usize;
    let s2_size = u16::from_le_bytes([data[2], data[3]]) as usize;
    let s3_size = u16::from_le_bytes([data[4], data[5]]) as usize;
    let s_data = &data[6..];

    let s1_end = s1_size;
    let s2_end = s1_end + s2_size;
    let s3_end = s2_end + s3_size;
    if s3_end > s_data.len() {
        return Err(io::Error::new(io::ErrorKind::InvalidData, "zstd: 4-stream jump table overflows data"));
    }

    let seg_size = (regen_size + 3) / 4;
    let sizes = [
        seg_size.min(regen_size),
        seg_size.min(regen_size.saturating_sub(seg_size)),
        seg_size.min(regen_size.saturating_sub(seg_size * 2)),
        regen_size.saturating_sub(seg_size * 3),
    ];

    let streams: [&[u8]; 4] = [
        &s_data[..s1_end],
        &s_data[s1_end..s2_end],
        &s_data[s2_end..s3_end],
        &s_data[s3_end..],
    ];

    let mut output = Vec::with_capacity(regen_size);
    for i in 0..4 {
        let decoded = decode_literals_1stream(table, streams[i], sizes[i])?;
        output.extend_from_slice(&decoded);
    }
    Ok(output)
}

#[inline(always)]
fn highest_bit(v: u32) -> u32 {
    debug_assert!(v > 0);
    31 - v.leading_zeros()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_huffman_decode_text100() {
        // FSE-compressed Huffman weights from text_100 level1 zstd stream.
        let fse_weight_data = [
            0x80, 0xa9, 0x6d, 0xc0, 0x7f, 0xaf, 0x2c, 0xb6, 0x50, 0xf9, 0x71, 0xad,
            0x61, 0x52, 0x22, 0x8d, 0xc2, 0x04, 0x3e, 0x83, 0x0b, 0x33, 0xa5, 0x01,
        ];
        // Huffman bitstream (56 bytes) for decoding 99 literal bytes.
        let huf_stream = [
            0x77, 0x6e, 0x93, 0xce, 0xf4, 0xd4, 0x15, 0xe2, 0x5c, 0xfb, 0xb5, 0x6b,
            0x42, 0x33, 0x60, 0x8f, 0x88, 0xf7, 0xd4, 0x89, 0x07, 0x8b, 0xd1, 0x52,
            0x75, 0x46, 0x4f, 0x14, 0x38, 0x60, 0xb0, 0x98, 0xd1, 0x20, 0xc7, 0x3f,
            0xd6, 0xf9, 0x04, 0x5a, 0x15, 0x5a, 0x06, 0x2f, 0xb0, 0xe0, 0xa4, 0xc0,
            0x24, 0xf8, 0x60, 0xb7, 0x10, 0xfe, 0x31, 0x08,
        ];

        // Step 1: Decode weights from FSE data.
        let weights = decode_weights_fse(&fse_weight_data).unwrap();
        eprintln!("Decoded {} weights: {:?}", weights.len(), &weights);

        // Step 2: Build Huffman table.
        let table = HufTable::from_weights(&weights).unwrap();
        eprintln!("Huffman table: max_bits={}, table_size={}", table.max_bits, table.table.len());

        // Step 3: Decode literals.
        let literals = decode_literals_1stream(&table, &huf_stream, 99).unwrap();
        eprintln!("Decoded {} literals, first 20: {:?}", literals.len(), &literals[..20.min(literals.len())]);
        eprintln!("As string: {:?}", std::str::from_utf8(&literals[..20.min(literals.len())]));

        let expected = b"The quick brown fox jumps over the lazy dog. Lorem ipsum dolor sit amet, consectetur adipiscing eli";
        assert_eq!(literals, expected, "literals mismatch");
    }
}
