//! .xz file format framing.
//!
//! Spec: <https://tukaani.org/xz/xz-file-format.txt>
//!
//! Layout of a `.xz` file:
//!
//! ```text
//! Stream Header (12 bytes)
//!   Header Magic (6) = FD 37 7A 58 5A 00
//!   Stream Flags (2) = 00 <check_id>
//!   CRC32 over Stream Flags (4)
//!
//! Block 1
//!   Block Header
//!     Header Size byte                       (1)
//!     Block Flags                            (1)
//!     [Compressed Size]                      (multibyte int, optional)
//!     [Uncompressed Size]                    (multibyte int, optional)
//!     Filter Flags x N                       (filter ID + properties)
//!     Header Padding                         (zero pad to 4-byte mult)
//!     CRC32                                  (4)
//!   Compressed Data                          (LZMA2 chunk stream)
//!   Block Padding                            (zero pad to 4-byte mult)
//!   Check                                    (0/4/8/32 bytes)
//!
//! Block 2
//! ...
//!
//! Index
//!   Index Indicator = 0x00                   (1)
//!   Number of Records (multibyte int)        (1+)
//!   For each block: Unpadded Size + Uncompressed Size  (multibyte int x 2)
//!   Index Padding                            (zero pad to 4-byte mult)
//!   CRC32 of Index                           (4)
//!
//! Stream Footer (12 bytes)
//!   CRC32 over Backward Size + Stream Flags  (4)
//!   Backward Size                            (4)
//!   Stream Flags                             (2)  must equal Stream Header's
//!   Footer Magic = 59 5A                     (2)
//! ```

use std::io;

use super::check::{crc32, crc64};
use super::lzma2::decode_lzma2;
use super::options::Check;

const HEADER_MAGIC: [u8; 6] = [0xFD, 0x37, 0x7A, 0x58, 0x5A, 0x00];
const FOOTER_MAGIC: [u8; 2] = [0x59, 0x5A];

// =========================================================================
// Multibyte integer encoding (LEB128-style, max 9 bytes per spec)
// =========================================================================

fn read_multibyte_int(input: &[u8], pos: &mut usize) -> io::Result<u64> {
    let mut value: u64 = 0;
    let start = *pos;
    for i in 0..9 {
        if *pos >= input.len() {
            return Err(io::Error::new(
                io::ErrorKind::UnexpectedEof,
                "xz: truncated multibyte integer",
            ));
        }
        let b = input[*pos];
        *pos += 1;
        value |= ((b & 0x7F) as u64) << (i * 7);
        if b & 0x80 == 0 {
            // Spec: a multi-byte int must use the minimum number of bytes
            // (no trailing zeros).
            if i > 0 && b == 0 {
                let _ = start;
                return Err(io::Error::new(
                    io::ErrorKind::InvalidData,
                    "xz: multibyte integer not minimally encoded",
                ));
            }
            return Ok(value);
        }
    }
    Err(io::Error::new(
        io::ErrorKind::InvalidData,
        "xz: multibyte integer too long",
    ))
}

fn write_multibyte_int(out: &mut Vec<u8>, mut value: u64) {
    while value >= 0x80 {
        out.push(((value & 0x7F) as u8) | 0x80);
        value >>= 7;
    }
    out.push(value as u8);
}

// =========================================================================
// Stream-level decode
// =========================================================================

pub fn decode_xz_stream(input: &[u8], output: &mut Vec<u8>) -> io::Result<usize> {
    if input.len() < 12 + 12 {
        return Err(io::Error::new(
            io::ErrorKind::UnexpectedEof,
            "xz: file too small to contain even an empty stream",
        ));
    }

    let mut pos = 0usize;
    let mut total_consumed = 0usize;

    while pos < input.len() {
        // Permit a sequence of concatenated streams (per the spec).
        // Skip any 4-byte-aligned stream padding zeros between streams.
        while pos < input.len() && input[pos] == 0 {
            pos += 1;
        }
        if pos == input.len() {
            break;
        }

        // ----- Stream Header -----
        if input.len() - pos < 12 {
            return Err(io::Error::new(
                io::ErrorKind::UnexpectedEof,
                "xz: short stream header",
            ));
        }
        if input[pos..pos + 6] != HEADER_MAGIC {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "xz: bad stream header magic",
            ));
        }
        let stream_flags = &input[pos + 6..pos + 8];
        if stream_flags[0] != 0 {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "xz: stream flags reserved byte must be zero",
            ));
        }
        let check = Check::from_id(stream_flags[1] & 0x0F)?;
        let header_crc = u32::from_le_bytes([
            input[pos + 8],
            input[pos + 9],
            input[pos + 10],
            input[pos + 11],
        ]);
        let computed = crc32(stream_flags);
        if computed != header_crc {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "xz: stream header CRC mismatch",
            ));
        }
        pos += 12;
        let stream_start = pos;

        // ----- Blocks -----
        let mut blocks: Vec<(u64, u64)> = Vec::new(); // (unpadded_size, uncompressed_size)
        loop {
            if pos >= input.len() {
                return Err(io::Error::new(
                    io::ErrorKind::UnexpectedEof,
                    "xz: stream ended without index",
                ));
            }
            if input[pos] == 0x00 {
                // Index indicator — block sequence done.
                break;
            }
            let (unpadded_size, uncompressed_size) =
                decode_block(input, &mut pos, check, output)?;
            blocks.push((unpadded_size, uncompressed_size));
        }

        // ----- Index -----
        let index_start = pos;
        // Index indicator already at input[pos] == 0x00.
        pos += 1;
        let num_records = read_multibyte_int(input, &mut pos)?;
        if num_records as usize != blocks.len() {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "xz: index record count doesn't match block count",
            ));
        }
        for &(want_unpadded, want_uncomp) in &blocks {
            let unpadded = read_multibyte_int(input, &mut pos)?;
            let uncomp = read_multibyte_int(input, &mut pos)?;
            if unpadded != want_unpadded || uncomp != want_uncomp {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidData,
                    "xz: index record doesn't match decoded block",
                ));
            }
        }
        // Index padding to 4-byte multiple.
        while (pos - index_start) % 4 != 0 {
            if pos >= input.len() || input[pos] != 0 {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidData,
                    "xz: bad index padding",
                ));
            }
            pos += 1;
        }
        // Index CRC32.
        if input.len() - pos < 4 {
            return Err(io::Error::new(
                io::ErrorKind::UnexpectedEof,
                "xz: missing index CRC",
            ));
        }
        let index_bytes = &input[index_start..pos];
        let stored_index_crc = u32::from_le_bytes([
            input[pos],
            input[pos + 1],
            input[pos + 2],
            input[pos + 3],
        ]);
        if crc32(index_bytes) != stored_index_crc {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "xz: index CRC mismatch",
            ));
        }
        pos += 4;
        let index_size = (pos - index_start) as u64;

        // ----- Stream Footer -----
        if input.len() - pos < 12 {
            return Err(io::Error::new(
                io::ErrorKind::UnexpectedEof,
                "xz: short stream footer",
            ));
        }
        let footer_crc = u32::from_le_bytes([
            input[pos],
            input[pos + 1],
            input[pos + 2],
            input[pos + 3],
        ]);
        let backward_size_raw = u32::from_le_bytes([
            input[pos + 4],
            input[pos + 5],
            input[pos + 6],
            input[pos + 7],
        ]);
        let footer_flags = &input[pos + 8..pos + 10];
        if input[pos + 10..pos + 12] != FOOTER_MAGIC {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "xz: bad stream footer magic",
            ));
        }
        if footer_flags != stream_flags {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "xz: footer stream flags don't match header",
            ));
        }
        // Footer CRC covers backward_size + stream_flags (6 bytes).
        let mut crc_buf = [0u8; 6];
        crc_buf[0..4].copy_from_slice(&backward_size_raw.to_le_bytes());
        crc_buf[4..6].copy_from_slice(footer_flags);
        if crc32(&crc_buf) != footer_crc {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "xz: stream footer CRC mismatch",
            ));
        }
        let backward_size = (backward_size_raw as u64 + 1) * 4;
        if backward_size != index_size {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "xz: footer backward_size doesn't match index size",
            ));
        }
        pos += 12;
        total_consumed = pos;
        let _ = stream_start;
    }
    Ok(total_consumed)
}

/// Decode a single block.  On entry `pos` points at the block header
/// size byte; on exit it points just past the block check.
fn decode_block(
    input: &[u8],
    pos: &mut usize,
    check: Check,
    output: &mut Vec<u8>,
) -> io::Result<(u64, u64)> {
    let block_start = *pos;
    let block_header_size_byte = input[*pos];
    if block_header_size_byte == 0 {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "xz: block header size byte is zero (= index indicator)",
        ));
    }
    let block_header_size = (block_header_size_byte as usize + 1) * 4;
    if *pos + block_header_size > input.len() {
        return Err(io::Error::new(
            io::ErrorKind::UnexpectedEof,
            "xz: block header truncated",
        ));
    }
    let header = &input[*pos..*pos + block_header_size];
    // Last 4 bytes are CRC32 over header[0..len-4].
    let header_crc =
        u32::from_le_bytes(header[block_header_size - 4..].try_into().unwrap());
    if crc32(&header[..block_header_size - 4]) != header_crc {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "xz: block header CRC mismatch",
        ));
    }

    let mut hp = 1usize; // header parser cursor (skip size byte)
    let block_flags = header[hp];
    hp += 1;
    let num_filters = (block_flags & 0x03) + 1;
    let has_compressed_size = (block_flags & 0x40) != 0;
    let has_uncompressed_size = (block_flags & 0x80) != 0;
    if block_flags & 0x3C != 0 {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "xz: block flags reserved bits must be zero",
        ));
    }

    let declared_compressed_size = if has_compressed_size {
        let mut h = hp;
        let v = read_multibyte_int(header, &mut h)?;
        hp = h;
        Some(v)
    } else {
        None
    };
    let declared_uncompressed_size = if has_uncompressed_size {
        let mut h = hp;
        let v = read_multibyte_int(header, &mut h)?;
        hp = h;
        Some(v)
    } else {
        None
    };

    // Filter flags.  We only support a single LZMA2 filter for now.
    if num_filters != 1 {
        return Err(io::Error::new(
            io::ErrorKind::Unsupported,
            format!(
                "xz: filter chains with {} filters not yet supported",
                num_filters
            ),
        ));
    }
    let mut h = hp;
    let filter_id = read_multibyte_int(header, &mut h)?;
    if filter_id != 0x21 {
        return Err(io::Error::new(
            io::ErrorKind::Unsupported,
            format!("xz: unsupported filter id 0x{:x} (only LZMA2 = 0x21 is supported)", filter_id),
        ));
    }
    let props_size = read_multibyte_int(header, &mut h)? as usize;
    if props_size != 1 {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "xz: LZMA2 properties size must be 1",
        ));
    }
    if h + 1 > header.len() {
        return Err(io::Error::new(
            io::ErrorKind::UnexpectedEof,
            "xz: missing LZMA2 dict_size byte",
        ));
    }
    let dict_size_byte = header[h];
    h += 1;
    let dict_size = decode_lzma2_dict_size(dict_size_byte)?;
    hp = h;

    // Header padding zeros up to block_header_size - 4.
    while hp < block_header_size - 4 {
        if header[hp] != 0 {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "xz: block header padding must be zero",
            ));
        }
        hp += 1;
    }

    *pos += block_header_size;

    // ----- Compressed payload (LZMA2) -----
    let unc_start = output.len();
    let payload_start = *pos;
    let consumed = decode_lzma2(&input[*pos..], dict_size, output)?;
    *pos += consumed;
    let produced = output.len() - unc_start;

    if let Some(want) = declared_compressed_size {
        if (consumed as u64) != want {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!(
                    "xz: block compressed size mismatch (declared {}, actual {})",
                    want, consumed
                ),
            ));
        }
    }
    if let Some(want) = declared_uncompressed_size {
        if (produced as u64) != want {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!(
                    "xz: block uncompressed size mismatch (declared {}, actual {})",
                    want, produced
                ),
            ));
        }
    }

    // Block padding to 4-byte multiple, measured from block_start.
    while (*pos - block_start) % 4 != 0 {
        if *pos >= input.len() || input[*pos] != 0 {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "xz: bad block padding",
            ));
        }
        *pos += 1;
    }

    // Check.
    let check_size = check.size();
    if input.len() - *pos < check_size {
        return Err(io::Error::new(
            io::ErrorKind::UnexpectedEof,
            "xz: missing block check",
        ));
    }
    verify_check(check, &output[unc_start..], &input[*pos..*pos + check_size])?;
    *pos += check_size;

    // Unpadded size = block header + compressed data + check (NO padding).
    let unpadded_size = block_header_size as u64 + consumed as u64 + check_size as u64;
    let _ = payload_start;
    Ok((unpadded_size, produced as u64))
}

fn verify_check(check: Check, plain: &[u8], stored: &[u8]) -> io::Result<()> {
    match check {
        Check::None => Ok(()),
        Check::Crc32 => {
            let computed = crc32(plain).to_le_bytes();
            if computed != stored {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidData,
                    "xz: block CRC32 mismatch",
                ));
            }
            Ok(())
        }
        Check::Crc64 => {
            let computed = crc64(plain).to_le_bytes();
            if computed != stored {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidData,
                    "xz: block CRC64 mismatch",
                ));
            }
            Ok(())
        }
        Check::Sha256 => {
            // We don't ship a SHA-256 implementation in this MVP; the
            // .xz default is CRC64 anyway.  Reject.
            Err(io::Error::new(
                io::ErrorKind::Unsupported,
                "xz: SHA-256 check not yet supported",
            ))
        }
    }
}

/// Decode the LZMA2 1-byte dict_size encoding (per liblzma spec):
///
/// ```text
/// 0..=39   dict_size = (2 | (b & 1)) << (b/2 + 11)
/// 40       dict_size = u32::MAX
/// ```
pub(crate) fn decode_lzma2_dict_size(b: u8) -> io::Result<u32> {
    if b > 40 {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "xz: invalid LZMA2 dict_size byte",
        ));
    }
    if b == 40 {
        Ok(u32::MAX)
    } else {
        let dict = (2u32 | (b as u32 & 1)) << (b as u32 / 2 + 11);
        Ok(dict)
    }
}

// =========================================================================
// Stream-level encode
// =========================================================================

/// Encode `input` as a complete .xz stream at the given preset and check.
/// Convenience wrapper for callers that don't need fine-grained
/// `LzmaOptions` control.
pub fn encode_xz_stream(
    input: &[u8],
    preset: u32,
    check: Check,
    output: &mut Vec<u8>,
) -> io::Result<()> {
    let opts = super::options::LzmaOptions::new_preset(preset)?;
    encode_xz_stream_with_options(input, &opts, check, output)
}

/// Encode `input` as a complete .xz stream using the given fully-resolved
/// LZMA options + check.  Currently emits a single block with a single
/// LZMA2 filter.
pub fn encode_xz_stream_with_options(
    input: &[u8],
    opts: &super::options::LzmaOptions,
    check: Check,
    output: &mut Vec<u8>,
) -> io::Result<()> {
    // ----- Stream Header -----
    output.extend_from_slice(&HEADER_MAGIC);
    let stream_flags = [0u8, check.id() & 0x0F];
    output.extend_from_slice(&stream_flags);
    output.extend_from_slice(&crc32(&stream_flags).to_le_bytes());

    // Encode the LZMA2 payload first; we need its size for the index.
    let mut payload = Vec::new();
    super::lzma::encode_lzma_to_lzma2(input, opts, &mut payload)?;

    // ----- Block Header -----
    let dict_size_byte = encode_lzma2_dict_size(opts.dict_size);
    let block_header_start = output.len();
    let mut header = Vec::new();
    // Block flags: 0 filters - 1 = 0, no comp/unc size present.
    header.push(0u8); // we'll fill in the size byte after we know the size
    header.push(0u8); // block flags = 0 (1 filter, no sizes)
    // Filter Flags for LZMA2: filter_id = 0x21, props_size = 1, props = dict_size byte.
    write_multibyte_int(&mut header, 0x21);
    write_multibyte_int(&mut header, 1);
    header.push(dict_size_byte);
    // Pad to 4-byte multiple AFTER reserving 4 bytes for the trailing CRC.
    let unpadded_no_crc = header.len() + 4;
    let padded = (unpadded_no_crc + 3) & !3;
    while header.len() + 4 < padded {
        header.push(0);
    }
    let block_header_size = padded; // includes the trailing 4-byte CRC
    let size_byte = (block_header_size / 4 - 1) as u8;
    header[0] = size_byte;
    let crc = crc32(&header);
    header.extend_from_slice(&crc.to_le_bytes());
    debug_assert_eq!(header.len(), block_header_size);
    output.extend_from_slice(&header);

    // ----- Compressed payload -----
    let payload_start = output.len();
    output.extend_from_slice(&payload);
    let payload_size = output.len() - payload_start;

    // Block padding.
    while (output.len() - block_header_start) % 4 != 0 {
        output.push(0);
    }

    // Block check.
    match check {
        Check::None => {}
        Check::Crc32 => {
            let v = crc32(input).to_le_bytes();
            output.extend_from_slice(&v);
        }
        Check::Crc64 => {
            let v = crc64(input).to_le_bytes();
            output.extend_from_slice(&v);
        }
        Check::Sha256 => {
            return Err(io::Error::new(
                io::ErrorKind::Unsupported,
                "xz: SHA-256 check not yet supported",
            ));
        }
    }

    let unpadded_size = block_header_size as u64 + payload_size as u64 + check.size() as u64;
    let uncompressed_size = input.len() as u64;

    // ----- Index -----
    let index_start = output.len();
    output.push(0x00); // index indicator
    write_multibyte_int(output, 1); // 1 record
    write_multibyte_int(output, unpadded_size);
    write_multibyte_int(output, uncompressed_size);
    while (output.len() - index_start) % 4 != 0 {
        output.push(0);
    }
    let index_bytes = &output[index_start..];
    let index_crc = crc32(index_bytes);
    output.extend_from_slice(&index_crc.to_le_bytes());
    let index_size = output.len() - index_start;
    debug_assert!(index_size % 4 == 0);

    // ----- Stream Footer -----
    let backward_size_raw = (index_size / 4 - 1) as u32;
    let mut footer_crc_buf = [0u8; 6];
    footer_crc_buf[0..4].copy_from_slice(&backward_size_raw.to_le_bytes());
    footer_crc_buf[4..6].copy_from_slice(&stream_flags);
    let footer_crc = crc32(&footer_crc_buf);
    output.extend_from_slice(&footer_crc.to_le_bytes());
    output.extend_from_slice(&backward_size_raw.to_le_bytes());
    output.extend_from_slice(&stream_flags);
    output.extend_from_slice(&FOOTER_MAGIC);

    Ok(())
}

/// Encode an LZMA2 dict_size byte (inverse of `decode_lzma2_dict_size`).
fn encode_lzma2_dict_size(dict_size: u32) -> u8 {
    if dict_size == u32::MAX {
        return 40;
    }
    // Find the smallest b such that (2 | (b & 1)) << (b/2 + 11) >= dict_size.
    for b in 0..40u8 {
        let v = (2u32 | (b as u32 & 1)) << (b as u32 / 2 + 11);
        if v >= dict_size {
            return b;
        }
    }
    40
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn multibyte_round_trip() {
        for v in [0u64, 1, 0x7F, 0x80, 0x3FFF, 0x4000, 0xFFFFFFFF, 0x123456789ABC] {
            let mut buf = Vec::new();
            write_multibyte_int(&mut buf, v);
            let mut pos = 0;
            assert_eq!(read_multibyte_int(&buf, &mut pos).unwrap(), v);
            assert_eq!(pos, buf.len());
        }
    }

    #[test]
    fn dict_size_byte_round_trip() {
        for b in 0..=40u8 {
            let v = decode_lzma2_dict_size(b).unwrap();
            let back = encode_lzma2_dict_size(v);
            assert_eq!(b, back, "round-trip dict_size byte {} -> {} -> {}", b, v, back);
        }
    }
}
