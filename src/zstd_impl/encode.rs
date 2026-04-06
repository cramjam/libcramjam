//! Zstandard frame encoder (RFC 8878).
//!
//! Produces valid zstd frames. For initial implementation, uses raw (uncompressed)
//! blocks to ensure correctness, with compressed blocks to follow.

use std::io;

const ZSTD_MAGIC: u32 = 0xFD2FB528;

/// Compress `input` into a zstd frame.
///
/// `level`: 0 = store only, 1+ = compressed.
/// `content_size`: if Some, written into the frame header for decoder optimization.
pub fn encode_frame(input: &[u8], level: i32, content_size: Option<u64>) -> Vec<u8> {
    let mut out = Vec::with_capacity(input.len() + 64);

    // Frame header.
    out.extend_from_slice(&ZSTD_MAGIC.to_le_bytes());

    // Frame_Header_Descriptor.
    // Bits: FCS_Field_Size(2) | Single_Segment(1) | Unused(1) | Reserved(1) |
    //       Content_Checksum(1) | Dict_ID_Flag(2)
    let content_checksum = true;
    let single_segment = true; // No window descriptor needed.
    let fcs_field_size = fcs_field_size_for(input.len() as u64);

    let descriptor: u8 = (fcs_field_size.flag << 6)
        | ((single_segment as u8) << 5)
        | ((content_checksum as u8) << 2);
    out.push(descriptor);

    // Frame_Content_Size (variable length based on fcs_field_size).
    write_fcs(&mut out, input.len() as u64, fcs_field_size.bytes);

    // Blocks.
    if level == 0 || input.is_empty() {
        encode_raw_blocks(&mut out, input);
    } else {
        encode_raw_blocks(&mut out, input); // TODO: compressed blocks
    }

    // Content checksum (xxhash64 lower 32 bits).
    if content_checksum {
        let hash = super::decode::xxhash64_public(input, 0) as u32;
        out.extend_from_slice(&hash.to_le_bytes());
    }

    out
}

/// Worst-case compressed size (raw blocks + frame overhead).
pub fn compress_bound(input_len: usize) -> usize {
    // Frame header (max ~14 bytes) + block headers (3 bytes per 128KB block) + data + checksum (4).
    let num_blocks = (input_len + 131071) / 131072; // max block size = 128KB
    14 + num_blocks * 3 + input_len + 4
}

// ---------------------------------------------------------------------------
// Raw (uncompressed) blocks
// ---------------------------------------------------------------------------

const MAX_BLOCK_SIZE: usize = 128 * 1024; // 128 KB per RFC

fn encode_raw_blocks(out: &mut Vec<u8>, input: &[u8]) {
    let mut offset = 0;
    while offset < input.len() {
        let chunk = (input.len() - offset).min(MAX_BLOCK_SIZE);
        let last = offset + chunk >= input.len();

        // Block header: 3 bytes LE. Bits: Block_Size(21) | Block_Type(2) | Last_Block(1)
        let header = ((chunk as u32) << 3) | (0 << 1) | (last as u32); // type 0 = raw
        out.push(header as u8);
        out.push((header >> 8) as u8);
        out.push((header >> 16) as u8);
        out.extend_from_slice(&input[offset..offset + chunk]);

        offset += chunk;
    }

    if input.is_empty() {
        // Empty frame: single empty last block.
        let header: u32 = (0 << 3) | (0 << 1) | 1; // size=0, type=raw, last=true
        out.push(header as u8);
        out.push((header >> 8) as u8);
        out.push((header >> 16) as u8);
    }
}

// ---------------------------------------------------------------------------
// Frame Content Size encoding
// ---------------------------------------------------------------------------

struct FcsInfo {
    flag: u8,  // 2-bit flag for frame header descriptor
    bytes: u8, // number of bytes to write
}

fn fcs_field_size_for(size: u64) -> FcsInfo {
    if size <= 255 {
        FcsInfo { flag: 0, bytes: 1 } // single_segment + FCS=0 → 1 byte
    } else if size <= 65535 + 256 {
        FcsInfo { flag: 1, bytes: 2 }
    } else if size <= u32::MAX as u64 {
        FcsInfo { flag: 2, bytes: 4 }
    } else {
        FcsInfo { flag: 3, bytes: 8 }
    }
}

fn write_fcs(out: &mut Vec<u8>, size: u64, bytes: u8) {
    match bytes {
        1 => out.push(size as u8),
        2 => out.extend_from_slice(&((size - 256) as u16).to_le_bytes()),
        4 => out.extend_from_slice(&(size as u32).to_le_bytes()),
        8 => out.extend_from_slice(&size.to_le_bytes()),
        _ => {}
    }
}
