//! Legacy LZMA "Alone" / `.lzma` format.
//!
//! Layout:
//!
//! ```text
//! +------+--------+--------+----------+
//! |props | dict   | unc    | LZMA     |
//! | 1B   |  4B LE |  8B LE | stream...|
//! +------+--------+--------+----------+
//! ```
//!
//! * `props`  — LZMA properties byte: `(pb * 5 + lp) * 9 + lc` (max 224)
//! * `dict`   — dictionary size in bytes (little-endian u32)
//! * `unc`    — uncompressed size in bytes (little-endian u64).  The
//!              special value `u64::MAX` means "unknown size; decode
//!              until end of input".
//! * Followed by a single raw LZMA range-coded stream (NOT LZMA2).
//!
//! The format is what Python's `lzma.compress(..., format=FORMAT_ALONE)`
//! produces and what `lzma.decompress` reads when sniffing.

use std::io;

use super::lzma::{LzmaDecoder, LZMA_LCLP_MAX, LZMA_PB_MAX};
use super::range_coder::RangeDecoder;

pub const ALONE_HEADER_LEN: usize = 13;

/// Decode a legacy `.lzma` stream into `output`.
pub fn decode_alone(input: &[u8], output: &mut Vec<u8>) -> io::Result<()> {
    if input.len() < ALONE_HEADER_LEN {
        return Err(io::Error::new(
            io::ErrorKind::UnexpectedEof,
            "lzma-alone: input shorter than 13-byte header",
        ));
    }

    // 1) Properties byte → (lc, lp, pb)
    let props = input[0];
    let (lc, lp, pb) = decode_alone_props(props)?;

    // 2) Dictionary size (LE u32).
    let dict_size = u32::from_le_bytes([input[1], input[2], input[3], input[4]]);
    // liblzma rounds tiny dict sizes up to 4 KiB.
    let dict_size = dict_size.max(4096);

    // 3) Uncompressed size (LE u64).  u64::MAX means "unknown".
    let unc_raw = u64::from_le_bytes([
        input[5], input[6], input[7], input[8],
        input[9], input[10], input[11], input[12],
    ]);
    let known_size = unc_raw != u64::MAX;
    let target_size = if known_size { Some(unc_raw as usize) } else { None };

    // 4) Decode the LZMA payload that follows the 13-byte header.
    // Pad the payload so the range decoder's unchecked refill can never
    // read past the end of the input (a corrupt stream may try).
    let mut payload = input[ALONE_HEADER_LEN..].to_vec();
    payload.resize(payload.len() + 16, 0);
    let mut decoder = LzmaDecoder::new(lc, lp, pb, dict_size)?;
    decoder.dict_start = output.len();
    let mut rd = RangeDecoder::new(&payload)?;

    // The maximum we'll ever produce.  When the size is unknown we use
    // a generous cap (1 GiB) so the decoder loop has a budget; the
    // alone format also supports an end-of-payload distance marker
    // (`u32::MAX`) which `decode_to_dict` returns via `hit_marker`, so
    // we'll typically stop earlier than the cap.
    let cap = target_size.unwrap_or(1usize << 30);

    // `decode_into` writes produced bytes directly into `output` (which
    // doubles as the dictionary), so payloads larger than the dict size
    // work correctly even when the size header is "unknown".
    let (produced, hit_marker) = decoder.decode_into(&mut rd, cap, output)?;

    if let Some(want) = target_size {
        if produced != want {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!(
                    "lzma-alone: declared uncompressed size {} but produced {}",
                    want, produced
                ),
            ));
        }
    } else if !hit_marker {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "lzma-alone: stream lacks both an uncompressed-size header AND \
             an end-of-payload marker — can't tell where to stop",
        ));
    }

    Ok(())
}

/// Decode the LZMA properties byte: `(pb * 5 + lp) * 9 + lc`.
fn decode_alone_props(props: u8) -> io::Result<(u32, u32, u32)> {
    if props >= 9 * 5 * 5 {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "lzma-alone: invalid properties byte",
        ));
    }
    let lc = (props % 9) as u32;
    let rest = props / 9;
    let lp = (rest % 5) as u32;
    let pb = (rest / 5) as u32;
    if lc > LZMA_LCLP_MAX || lp > LZMA_LCLP_MAX || lc + lp > LZMA_LCLP_MAX || pb > LZMA_PB_MAX {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "lzma-alone: invalid lc/lp/pb in properties byte",
        ));
    }
    Ok((lc, lp, pb))
}

/// Sniff the leading bytes to decide whether this is XZ or ALONE.
///
/// XZ files start with the 6-byte magic `FD 37 7A 58 5A 00`.  Anything
/// else with a valid LZMA properties byte (`< 225`) AND a sane
/// dictionary size is treated as ALONE.
pub fn looks_like_alone(input: &[u8]) -> bool {
    if input.len() < ALONE_HEADER_LEN {
        return false;
    }
    // Reject anything that starts with the .xz magic (handled separately).
    if input.len() >= 6 && input[..6] == [0xFD, 0x37, 0x7A, 0x58, 0x5A, 0x00] {
        return false;
    }
    // Properties byte must be a valid (lc, lp, pb) encoding.
    let props = input[0];
    if props >= 9 * 5 * 5 {
        return false;
    }
    // The dictionary size field is a sanity check: liblzma writes
    // dict sizes up to 1 GiB; we accept anything <= 1 << 30.  Larger
    // values almost certainly mean we're misreading random data.
    let dict_size = u32::from_le_bytes([input[1], input[2], input[3], input[4]]);
    if dict_size > (1u32 << 30) {
        return false;
    }
    true
}
