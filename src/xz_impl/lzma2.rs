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
    // We allocate the LzmaDecoder LAZILY — many .xz streams (e.g. random
    // / incompressible data) consist entirely of uncompressed LZMA2
    // chunks, in which case we never need to pay for the (potentially
    // multi-MiB) dict allocation.  The decoder is built on the first LZMA
    // chunk that supplies properties; if a later LZMA chunk follows
    // earlier uncompressed chunks, we seed the decoder's dict from the
    // tail of `output` so back-references still work.
    let mut decoder: Option<LzmaDecoder> = None;
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
            if let Some(d) = decoder.as_mut() {
                d.dict = Dict::new(dict_size as usize);
                d.reset_state();
            }
            pos = decode_uncompressed_chunk_to_output(input, pos, output)?;
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
            pos = decode_uncompressed_chunk_to_output(input, pos, output)?;
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
            // Build a fresh decoder.  If there's an existing decoder its
            // dict bytes carry over (assuming no dict reset above).  If
            // this is the FIRST LZMA chunk after some uncompressed
            // chunks, seed the new decoder's dict from the tail of
            // `output` so back-references work.
            let new_decoder = match decoder.take() {
                Some(prior) => {
                    let mut d = LzmaDecoder::new(lc, lp, pb, dict_size)?;
                    d.dict = prior.dict;
                    d
                }
                None => {
                    let mut d = LzmaDecoder::new(lc, lp, pb, dict_size)?;
                    seed_dict_from_output(&mut d.dict, output);
                    d
                }
            };
            decoder = Some(new_decoder);
        } else {
            let d = decoder.as_mut().ok_or_else(|| {
                io::Error::new(
                    io::ErrorKind::InvalidData,
                    "lzma2: LZMA chunk requires properties but none have been set",
                )
            })?;
            if need_props {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidData,
                    "lzma2: LZMA chunk requires properties but none have been set",
                ));
            }
            if state_reset {
                d.reset_state();
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

        let d = decoder.as_mut().expect("decoder must exist on LZMA chunk");
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
        // user output.  Bulk memcpy (one or two slices, depending on
        // whether the cyclic buffer wraps).
        let cap = d.dict.buf.len();
        let start = (prev_total as usize) % cap;
        if start + produced <= cap {
            output.extend_from_slice(&d.dict.buf[start..start + produced]);
        } else {
            let first = cap - start;
            output.extend_from_slice(&d.dict.buf[start..cap]);
            output.extend_from_slice(&d.dict.buf[..produced - first]);
        }
    }

    // Spec requires an explicit END marker; if we get here the input was
    // truncated.
    Err(io::Error::new(
        io::ErrorKind::UnexpectedEof,
        "lzma2: stream ended without end marker",
    ))
}

/// Decode an uncompressed LZMA2 chunk into `output` only.  Used when no
/// LzmaDecoder has been allocated yet (i.e. there are no LZMA chunks in
/// the stream so far) — skips the dict-allocation cost entirely.
fn decode_uncompressed_chunk_to_output(
    input: &[u8],
    mut pos: usize,
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
    output.extend_from_slice(&input[pos..pos + size]);
    Ok(pos + size)
}

/// Seed an LzmaDecoder's freshly-allocated `Dict` with the tail of
/// `output`, so that LZMA chunks following uncompressed chunks can do
/// back-references into the bytes the uncompressed chunks emitted.
fn seed_dict_from_output(dict: &mut Dict, output: &[u8]) {
    if output.is_empty() {
        return;
    }
    let cap = dict.buf.len();
    let take = output.len().min(cap);
    let src = &output[output.len() - take..];
    dict.buf[..take].copy_from_slice(src);
    dict.total = take as u64;
    // The dict's "current write position" should equal `take` so the next
    // push() lands at index `take % cap`.  `total` already encodes this.
}

/// Bulk-copy a byte slice into a `Dict`'s cyclic buffer, splitting at the
/// wrap point if necessary.  Replaces the per-byte `dict.push(b)` loop.
fn extend_dict(dict: &mut Dict, bytes: &[u8]) {
    let cap = dict.buf.len();
    let mut start = (dict.total as usize) % cap;
    let mut remaining = bytes;
    while !remaining.is_empty() {
        let space = cap - start;
        let take = space.min(remaining.len());
        dict.buf[start..start + take].copy_from_slice(&remaining[..take]);
        remaining = &remaining[take..];
        start = (start + take) % cap;
    }
    dict.total += bytes.len() as u64;
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
