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

use super::lzma::{LzmaDecoder, LZMA_LCLP_MAX, LZMA_PB_MAX};
use super::range_coder::RangeDecoder;


/// Decode an LZMA2 stream from `input` into `output`.  `dict_size` is taken
/// from the surrounding xz block header (one byte: see `lzma2_props_decode`
/// in liblzma).  Returns the number of input bytes consumed, including the
/// terminating END marker (0x00).
pub fn decode_lzma2(input: &[u8], dict_size: u32, output: &mut Vec<u8>) -> io::Result<usize> {
    // The output Vec doubles as the LZ dictionary: `dict_start` is the
    // output offset of the most recent dict reset, and the decoder never
    // references bytes before it. Uncompressed chunks therefore just
    // append to `output` — no mirroring into a separate window.
    //
    // The LzmaDecoder (probability model) is allocated LAZILY — many .xz
    // streams (e.g. random / incompressible data) consist entirely of
    // uncompressed LZMA2 chunks and never need it.
    let mut decoder: Option<LzmaDecoder> = None;
    let mut need_dict_reset = true;
    let mut need_props = true;
    let mut dict_start = output.len();

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
            dict_start = output.len();
            if let Some(d) = decoder.as_mut() {
                d.reset_state();
            }
            pos = decode_uncompressed_chunk(input, pos, output)?;
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
            pos = decode_uncompressed_chunk(input, pos, output)?;
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
            dict_start = output.len();
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
            // Fresh probability model (new props imply a state reset).
            // The dictionary lives in `output`, so nothing carries over.
            decoder = Some(LzmaDecoder::new(lc, lp, pb, dict_size)?);
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
        // Decode straight from `input`; `RangeDecoder::end` bounds the range
        // decoder to exactly this chunk's compressed bytes, so a corrupt
        // chunk can neither read past the input allocation nor bleed into
        // the following chunk (`Rc::normalize` saturates + flags overrun).
        let chunk_len = compressed_size as usize;
        let chunk_start = pos;
        pos += chunk_len;
        let rc_input: &[u8] = &input[chunk_start..pos];

        let d = decoder.as_mut().expect("decoder must exist on LZMA chunk");
        d.dict_start = dict_start;
        let mut rd = RangeDecoder::new(rc_input)?;
        // `decode_into` writes the produced bytes directly into `output`,
        // which is also the dictionary — chunks larger than the dict size
        // (LZMA2 allows up to 2 MiB per chunk) decode correctly.
        let (produced, hit_marker) =
            d.decode_into(&mut rd, uncompressed_size as usize, output)?;

        if rd.pos > chunk_len {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "lzma2: range-coded data overruns chunk compressed size",
            ));
        }
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
    }

    // Spec requires an explicit END marker; if we get here the input was
    // truncated.
    Err(io::Error::new(
        io::ErrorKind::UnexpectedEof,
        "lzma2: stream ended without end marker",
    ))
}

/// Decode an uncompressed LZMA2 chunk: the bytes are appended to `output`,
/// which is also the dictionary, so following LZMA chunks see them as
/// back-reference history automatically.
fn decode_uncompressed_chunk(
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
