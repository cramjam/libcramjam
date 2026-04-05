//! DEFLATE decompression (RFC 1951)

use std::io;

use super::bitreader::BitReader;
use super::huffman::HuffmanDecoder;
use super::tables;

/// Decompress a raw DEFLATE stream.
///
/// Returns `(decompressed_data, input_bytes_consumed)`.
pub fn inflate(input: &[u8]) -> io::Result<(Vec<u8>, usize)> {
    let mut reader = BitReader::new(input);
    let mut output: Vec<u8> = Vec::new();

    loop {
        let bfinal = reader.read_bits(1)?;
        let btype = reader.read_bits(2)?;

        match btype {
            0 => inflate_stored(&mut reader, &mut output)?,
            1 => inflate_fixed(&mut reader, &mut output)?,
            2 => inflate_dynamic(&mut reader, &mut output)?,
            _ => {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidData,
                    "invalid deflate block type 3",
                ))
            }
        }

        if bfinal == 1 {
            break;
        }
    }

    let consumed = reader.bytes_consumed();
    Ok((output, consumed))
}

// ---------------------------------------------------------------------------
// Block type 0: stored (no compression)
// ---------------------------------------------------------------------------

fn inflate_stored(reader: &mut BitReader, output: &mut Vec<u8>) -> io::Result<()> {
    let header = reader.read_bytes(4)?;
    let len = u16::from_le_bytes([header[0], header[1]]);
    let nlen = u16::from_le_bytes([header[2], header[3]]);

    if len != !nlen {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "stored block LEN/NLEN mismatch",
        ));
    }

    let data = reader.read_bytes(len as usize)?;
    output.extend_from_slice(data);
    Ok(())
}

// ---------------------------------------------------------------------------
// Block type 1: fixed Huffman codes
// ---------------------------------------------------------------------------

fn inflate_fixed(reader: &mut BitReader, output: &mut Vec<u8>) -> io::Result<()> {
    let lit_dec = HuffmanDecoder::from_lengths(&tables::fixed_literal_lengths())?;
    let dist_dec = HuffmanDecoder::from_lengths(&tables::fixed_distance_lengths())?;
    decode_block(reader, &lit_dec, &dist_dec, output)
}

// ---------------------------------------------------------------------------
// Block type 2: dynamic Huffman codes
// ---------------------------------------------------------------------------

fn inflate_dynamic(reader: &mut BitReader, output: &mut Vec<u8>) -> io::Result<()> {
    let hlit = reader.read_bits(5)? as usize + 257;
    let hdist = reader.read_bits(5)? as usize + 1;
    let hclen = reader.read_bits(4)? as usize + 4;

    // Read code-length code lengths.
    let mut cl_lengths = [0u8; 19];
    for i in 0..hclen {
        cl_lengths[tables::CODE_LENGTH_ORDER[i]] = reader.read_bits(3)? as u8;
    }
    let cl_dec = HuffmanDecoder::from_lengths(&cl_lengths)?;

    // Decode literal/length + distance code lengths.
    let total = hlit + hdist;
    let mut all_lengths: Vec<u8> = Vec::with_capacity(total);

    while all_lengths.len() < total {
        let sym = cl_dec.decode(reader)?;
        match sym {
            0..=15 => all_lengths.push(sym as u8),
            16 => {
                // Repeat previous length 3..6 times.
                let count = reader.read_bits(2)? as usize + 3;
                let prev = *all_lengths
                    .last()
                    .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidData, "code 16 with no previous length"))?;
                let count = count.min(total - all_lengths.len());
                for _ in 0..count {
                    all_lengths.push(prev);
                }
            }
            17 => {
                // Repeat 0 for 3..10 times.
                let count = reader.read_bits(3)? as usize + 3;
                let count = count.min(total - all_lengths.len());
                for _ in 0..count {
                    all_lengths.push(0);
                }
            }
            18 => {
                // Repeat 0 for 11..138 times.
                let count = reader.read_bits(7)? as usize + 11;
                let count = count.min(total - all_lengths.len());
                for _ in 0..count {
                    all_lengths.push(0);
                }
            }
            _ => {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidData,
                    "invalid code-length symbol",
                ))
            }
        }
    }

    let lit_dec = HuffmanDecoder::from_lengths(&all_lengths[..hlit])?;
    let dist_dec = HuffmanDecoder::from_lengths(&all_lengths[hlit..])?;
    decode_block(reader, &lit_dec, &dist_dec, output)
}

// ---------------------------------------------------------------------------
// Decode compressed data within a block
// ---------------------------------------------------------------------------

fn decode_block(
    reader: &mut BitReader,
    lit_dec: &HuffmanDecoder,
    dist_dec: &HuffmanDecoder,
    output: &mut Vec<u8>,
) -> io::Result<()> {
    loop {
        let sym = lit_dec.decode(reader)?;
        match sym {
            0..=255 => {
                output.push(sym as u8);
            }
            256 => {
                // End of block.
                return Ok(());
            }
            257..=285 => {
                let idx = (sym - 257) as usize;
                if idx >= tables::LENGTH_BASE.len() {
                    return Err(io::Error::new(
                        io::ErrorKind::InvalidData,
                        "invalid length code",
                    ));
                }
                let length = tables::LENGTH_BASE[idx] as usize
                    + reader.read_bits(tables::LENGTH_EXTRA[idx] as u32)? as usize;

                let dist_sym = dist_dec.decode(reader)?;
                if dist_sym as usize >= tables::DISTANCE_BASE.len() {
                    return Err(io::Error::new(
                        io::ErrorKind::InvalidData,
                        "invalid distance code",
                    ));
                }
                let distance = tables::DISTANCE_BASE[dist_sym as usize] as usize
                    + reader.read_bits(tables::DISTANCE_EXTRA[dist_sym as usize] as u32)? as usize;

                if distance > output.len() {
                    return Err(io::Error::new(
                        io::ErrorKind::InvalidData,
                        "distance too far back",
                    ));
                }

                // Copy (may overlap: distance < length is valid and creates repeating patterns).
                let start = output.len() - distance;
                for i in 0..length {
                    let byte = output[start + i];
                    output.push(byte);
                }
            }
            _ => {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidData,
                    "invalid literal/length symbol",
                ))
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_inflate_flate2_level1_text() {
        // 100 bytes of text compressed by flate2 at level 1 (verified with Python zlib).
        let compressed: Vec<u8> = vec![
            0x0d, 0xc9, 0xcb, 0x15, 0x84, 0x20, 0x0c, 0x05, 0xd0, 0x56, 0x5e, 0x01, 0x73, 0xa6,
            0x12, 0x97, 0x36, 0x80, 0x18, 0x35, 0x0a, 0x04, 0x93, 0xe0, 0xaf, 0x7a, 0xdd, 0xde,
            0xdb, 0x2f, 0x84, 0xbd, 0x71, 0xdc, 0x30, 0xa8, 0x9c, 0x05, 0x93, 0x5c, 0x58, 0x5b,
            0xae, 0x06, 0x39, 0x48, 0xe1, 0x5f, 0xa7, 0xf0, 0xdc, 0x18, 0x65, 0xfe, 0xa3, 0x13,
            0xa5, 0x0c, 0xae, 0xd6, 0xf2, 0x07, 0x49, 0x14, 0xc6, 0x8e, 0x90, 0xc9, 0x7f, 0x88,
            0x52, 0x8c, 0xa2, 0x93, 0x37, 0x45, 0x18, 0xb9, 0xb2, 0x45, 0x2e, 0x33, 0x28, 0xb1,
            0xbf,
        ];

        let expected = b"The quick brown fox jumps over the lazy dog. Lorem ipsum dolor sit amet, consectetur adipiscing elit";
        let (output, _) = inflate(&compressed).unwrap();
        assert_eq!(&output, expected);
    }
}
