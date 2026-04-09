//! LZMA2 — chunked LZMA used inside the .xz block payload.
//!
//! Format (per chunk):
//!
//! ```text
//! control byte:
//!   0x00              END marker
//!   0x01              Uncompressed chunk WITH dict reset
//!   0x02              Uncompressed chunk WITHOUT dict reset
//!   0x80..=0xFF       LZMA chunk (top 5 bits = high bits of uncompressed size)
//!     bits 5..6:
//!       00            Old props, no state reset
//!       01            Old props, state reset
//!       10            New props (state reset implied), no dict reset
//!       11            New props + dict reset
//!
//! For LZMA chunks the body is:
//!   uncompressed_size_high(5 bits in control) << 16
//!     | (next byte) << 8 | (next byte) + 1     -> total uncompressed_size
//!   compressed_size_high << 8 | compressed_size_low + 1   (16-bit)
//!   [if new props]  1 byte (lc/lp/pb encoded)
//!   <compressed bytes>
//!
//! For uncompressed chunks the body is:
//!   compressed_size_high << 8 | compressed_size_low + 1   (16-bit)
//!   <uncompressed bytes>
//! ```
//!
//! The first chunk in a stream MUST be a dict-reset chunk (control 0x01
//! for uncompressed, or `>= 0xE0` for LZMA — i.e. "new props + dict reset"
//! in liblzma's terms).

use std::io;

use super::lzma::{Dict, LzmaDecoder, LZMA_LCLP_MAX, LZMA_PB_MAX};
use super::range_coder::RangeDecoder;

/// Decode an LZMA2 stream from `input` into `output`.  `dict_size` is taken
/// from the surrounding xz block header (one byte: see `lzma2_props_decode`
/// in liblzma).  Returns the number of input bytes consumed, including the
/// terminating END marker (0x00).
pub fn decode_lzma2(input: &[u8], dict_size: u32, output: &mut Vec<u8>) -> io::Result<usize> {
    // Decoder is created lazily on the first LZMA chunk that carries
    // properties — until then we don't know lc/lp/pb.  An uncompressed
    // dict-reset chunk can come first; in that case we still need a dict
    // ready, so allocate one with default 3/0/2 properties.
    let mut decoder: Option<LzmaDecoder> = None;
    let mut dict = Dict::new(dict_size as usize);
    let mut need_dict_reset = true;
    let mut need_props = true;

    let mut pos = 0usize;
    while pos < input.len() {
        let control = input[pos];
        pos += 1;

        if control == 0x00 {
            // End marker.
            return Ok(pos);
        }

        if control == 0x01 {
            // Uncompressed dict-reset chunk.
            need_props = true;
            need_dict_reset = false;
            // Reset dict.
            dict = Dict::new(dict_size as usize);
            if let Some(d) = decoder.as_mut() {
                d.dict = Dict::new(dict_size as usize);
                d.reset_state();
            }
            pos = decode_uncompressed_chunk(input, pos, &mut dict, output)?;
            continue;
        }

        if control == 0x02 {
            // Uncompressed chunk, no reset.
            if need_dict_reset {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidData,
                    "lzma2: uncompressed chunk before dict reset",
                ));
            }
            pos = decode_uncompressed_chunk(input, pos, &mut dict, output)?;
            // Mirror our scratch dict into the LzmaDecoder's dict.
            if let Some(d) = decoder.as_mut() {
                d.dict = clone_dict(&dict);
            }
            continue;
        }

        if control < 0x80 {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!("lzma2: invalid control byte 0x{:02x}", control),
            ));
        }

        // ----- LZMA chunk -----
        // Top 5 bits = high bits of uncompressed_size; bits 5..6 = mode flags.
        let mode = (control >> 5) & 0x03; // 0..=3
        let unc_high = (control & 0x1F) as u32;

        if pos + 4 > input.len() {
            return Err(io::Error::new(
                io::ErrorKind::UnexpectedEof,
                "lzma2: short LZMA chunk header",
            ));
        }
        let unc_mid = input[pos] as u32;
        let unc_low = input[pos + 1] as u32;
        let comp_hi = input[pos + 2] as u32;
        let comp_lo = input[pos + 3] as u32;
        pos += 4;

        let uncompressed_size = ((unc_high << 16) | (unc_mid << 8) | unc_low) + 1;
        let compressed_size = ((comp_hi << 8) | comp_lo) + 1;

        // mode == 3 (control >= 0xE0) ⇒ new props + dict reset
        // mode == 2 (control >= 0xC0) ⇒ new props, no dict reset
        // mode == 1 (control >= 0xA0) ⇒ state reset, old props
        // mode == 0 (control >= 0x80) ⇒ no reset
        let new_props = mode >= 2;
        let dict_reset = mode == 3;
        let state_reset = mode >= 1;

        if dict_reset {
            need_dict_reset = false;
            need_props = true;
            dict = Dict::new(dict_size as usize);
            if let Some(d) = decoder.as_mut() {
                d.dict = Dict::new(dict_size as usize);
            }
        } else if need_dict_reset {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "lzma2: LZMA chunk before dict reset",
            ));
        }

        if new_props {
            need_props = false;
            if pos >= input.len() {
                return Err(io::Error::new(
                    io::ErrorKind::UnexpectedEof,
                    "lzma2: missing properties byte",
                ));
            }
            let props = input[pos];
            pos += 1;
            let (lc, lp, pb) = decode_props(props)?;
            // Build a fresh decoder; carry over the existing dict bytes.
            let prior = std::mem::replace(&mut dict, Dict::new(dict_size as usize));
            let mut d = LzmaDecoder::new(lc, lp, pb, dict_size)?;
            d.dict = prior;
            decoder = Some(d);
        } else {
            if need_props {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidData,
                    "lzma2: LZMA chunk requires properties but none have been set",
                ));
            }
            if state_reset {
                if let Some(d) = decoder.as_mut() {
                    d.reset_state();
                }
            }
            // Move our scratch dict into the existing decoder so it has the
            // latest bytes from any preceding uncompressed chunks.
            if let Some(d) = decoder.as_mut() {
                let bytes_to_copy = dict.total;
                if bytes_to_copy > d.dict.total {
                    d.dict = clone_dict(&dict);
                }
            }
        }

        // Decode the LZMA payload.
        if pos + compressed_size as usize > input.len() {
            return Err(io::Error::new(
                io::ErrorKind::UnexpectedEof,
                "lzma2: LZMA payload exceeds remaining input",
            ));
        }
        let chunk_in = &input[pos..pos + compressed_size as usize];
        pos += compressed_size as usize;

        let d = decoder.as_mut().expect("decoder must exist by now");
        let prev_total = d.dict.total;
        let mut rd = RangeDecoder::new(chunk_in)?;
        let (produced, hit_marker) = d.decode_to_dict(&mut rd, uncompressed_size as usize)?;

        if hit_marker {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "lzma2: end-of-payload marker not allowed inside LZMA2 chunks",
            ));
        }
        if produced != uncompressed_size as usize {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "lzma2: chunk produced wrong number of bytes",
            ));
        }

        // Append the bytes the decoder just wrote into its dict to the
        // user output, and mirror them into our scratch dict.
        let cap = d.dict.buf.len();
        for i in 0..produced {
            let p = ((prev_total as usize + i) % cap) as usize;
            let b = d.dict.buf[p];
            output.push(b);
            dict.push(b);
        }
    }

    // Spec requires an explicit END marker; if we get here the input was
    // truncated.
    Err(io::Error::new(
        io::ErrorKind::UnexpectedEof,
        "lzma2: stream ended without end marker",
    ))
}

fn decode_uncompressed_chunk(
    input: &[u8],
    mut pos: usize,
    dict: &mut Dict,
    output: &mut Vec<u8>,
) -> io::Result<usize> {
    if pos + 2 > input.len() {
        return Err(io::Error::new(
            io::ErrorKind::UnexpectedEof,
            "lzma2: short uncompressed chunk header",
        ));
    }
    let size_hi = input[pos] as usize;
    let size_lo = input[pos + 1] as usize;
    pos += 2;
    let size = ((size_hi << 8) | size_lo) + 1;
    if pos + size > input.len() {
        return Err(io::Error::new(
            io::ErrorKind::UnexpectedEof,
            "lzma2: uncompressed chunk truncated",
        ));
    }
    let bytes = &input[pos..pos + size];
    pos += size;
    output.extend_from_slice(bytes);
    for &b in bytes {
        dict.push(b);
    }
    Ok(pos)
}

/// Decode the LZMA properties byte: `(pb * 5 + lp) * 9 + lc`.
fn decode_props(props: u8) -> io::Result<(u32, u32, u32)> {
    if props >= 9 * 5 * 5 {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "lzma: invalid properties byte",
        ));
    }
    let lc = (props % 9) as u32;
    let rest = props / 9;
    let lp = (rest % 5) as u32;
    let pb = (rest / 5) as u32;
    if lc > LZMA_LCLP_MAX || lp > LZMA_LCLP_MAX || lc + lp > LZMA_LCLP_MAX || pb > LZMA_PB_MAX {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "lzma: invalid lc/lp/pb in properties byte",
        ));
    }
    Ok((lc, lp, pb))
}

/// Clone helper for `Dict` (the struct doesn't derive Clone to keep the
/// API surface small, but the LZMA2 driver needs to share dict bytes
/// across the scratch buffer used for uncompressed chunks).
fn clone_dict(d: &Dict) -> Dict {
    Dict {
        buf: d.buf.clone(),
        total: d.total,
    }
}
