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
    pub fn from_weights(weights: &[u8]) -> io::Result<Self> {
        if weights.is_empty() {
            return Err(io::Error::new(io::ErrorKind::InvalidData, "zstd: empty Huffman weights"));
        }

        // Determine max number of bits.
        let max_weight = *weights.iter().max().unwrap() as u32;
        if max_weight == 0 {
            return Err(io::Error::new(io::ErrorKind::InvalidData, "zstd: all Huffman weights are zero"));
        }

        // Sum of (1 << (max_weight - w)) for all symbols with w > 0 must be a power of 2.
        let weight_sum: u32 = weights
            .iter()
            .filter(|&&w| w > 0)
            .map(|&w| 1u32 << (w as u32 - 1))
            .sum();

        // max_bits = highest_bit(weight_sum) + 1, but weight_sum should be 2^(max_bits-1).
        let max_bits = highest_bit(weight_sum) + 1;
        if max_bits > HUF_MAX_TABLE_LOG {
            return Err(io::Error::new(io::ErrorKind::InvalidData, "zstd: Huffman table log too large"));
        }

        // Compute code lengths: num_bits[sym] = max_bits + 1 - weight[sym] (for weight > 0).
        let table_size = 1usize << max_bits;
        let mut table = vec![HufEntry::default(); table_size];

        // Assign codes using canonical Huffman ordering.
        // Sort symbols by weight (descending = shorter codes first).
        let mut symbols: Vec<(u8, u8)> = weights
            .iter()
            .enumerate()
            .filter(|(_, &w)| w > 0)
            .map(|(sym, &w)| (sym as u8, max_bits as u8 + 1 - w))
            .collect();
        symbols.sort_by_key(|&(_, bits)| bits);

        // Fill table with replicated entries (same approach as DEFLATE Huffman).
        let mut code = 0u32;
        let mut prev_bits = 0u8;
        for &(sym, bits) in &symbols {
            if bits != prev_bits {
                code <<= bits - prev_bits;
                prev_bits = bits;
            }
            let fill = 1usize << (max_bits as u8 - bits);
            for j in 0..fill {
                let idx = ((code as usize) << (max_bits as u8 - bits)) | j;
                // Reverse bits for the index (zstd Huffman reads MSB-first).
                // Actually, zstd Huffman codes are stored MSB-first and the table is
                // indexed by the raw bits as read from the MSB of the bitstream.
                // So the index IS the code in MSB order, padded with suffix bits.
                table[idx] = HufEntry { symbol: sym, num_bits: bits };
            }
            code += 1;
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
        let idx = bits.peek_bits(self.max_bits);
        let entry = self.table[idx as usize];
        bits.consume(entry.num_bits as u32);
        entry.symbol
    }
}

/// Decode Huffman weights using an FSE-compressed bitstream.
fn decode_weights_fse(data: &[u8]) -> io::Result<Vec<u8>> {
    let mut reader = super::bits::ForwardByteReader::new(data);
    let fse_table = fse::FseTable::decode_table(&mut reader, 255, 7)?;
    let remaining = &data[reader.position()..];

    let mut bits = ReverseBitReader::new(remaining)?;
    let mut state = bits.read_bits(fse_table.accuracy_log);

    let mut weights = Vec::new();
    loop {
        let sym = fse_table.symbol(state);
        weights.push(sym);
        if bits.is_done() {
            break;
        }
        state = fse_table.next_state(state, &mut bits);
        bits.reload();
    }
    Ok(weights)
}

/// Decode Huffman-compressed literals using 1 stream.
pub fn decode_literals_1stream(table: &HufTable, data: &[u8], regen_size: usize) -> io::Result<Vec<u8>> {
    let mut bits = ReverseBitReader::new(data)?;
    let mut output = Vec::with_capacity(regen_size);
    while output.len() < regen_size {
        bits.reload();
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
