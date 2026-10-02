//! Pure-Rust Zstandard implementation (RFC 8878).
//!
//! Decompression: our own decoder (`decode::decode_frame`).
//! Compression: our own native encoder (`encode::encode_frame`) — raw blocks
//! at level 0, LZ77 + predefined-FSE sequences at level >= 1.
// Every `unsafe {}` block in the pure-Rust codecs carries a `// SAFETY:`
// comment naming the invariant it relies on (checked by Miri in CI).
#![deny(clippy::undocumented_unsafe_blocks)]


mod bitc;
mod bits;
mod cparams;
mod entropy;
mod parse_fast;
mod parse_lazy;
mod seqstore;
mod decode;
pub mod encode;
mod fse;
mod huf;

use std::io::{self, Read, Write};

/// Default zstd compression level, matching C zstd's `ZSTD_defaultCLevel()` = 3.
pub const DEFAULT_COMPRESSION_LEVEL: i32 = 3;

/// Map a public (libzstd-semantics) level to the encoder's: 0 means the
/// default level and values above 22 clamp, as in `ZSTD_compressStream2`.
/// Negative ("fast") levels aren't implemented; they use level 1, the
/// fastest strategy we have (never the encoder's internal raw-block mode,
/// which would silently store data uncompressed).
pub(crate) fn api_level(level: i32) -> i32 {
    match level {
        0 => DEFAULT_COMPRESSION_LEVEL,
        l if l < 0 => 1, // ponytail: approximates libzstd's negative levels; add an acceleration knob if anyone needs them
        l => l.min(cparams::MAX_CLEVEL),
    }
}

/// Decompress a zstd frame.
pub fn decompress<W: Write + ?Sized, R: Read>(
    mut input: R,
    output: &mut W,
) -> io::Result<usize> {
    let mut sink = crate::SinkRef(output);
    crate::scratch_pair_with(|data, buf| {
        input.read_to_end(data)?;
        if data.is_empty() {
            // Not a stream; the C library rejects it too.
            return Err(io::Error::new(io::ErrorKind::UnexpectedEof, "zstd: empty input"));
        }
        let mut consumed = 0usize;
        let mut total = 0usize;
        while consumed < data.len() {
            let (n, produced) = decode::decode_frame_streaming(&data[consumed..], buf, Some(&mut sink))?;
            if n == 0 {
                break;
            }
            // Flush what the frame left in the scratch (streaming mode) or
            // the whole frame (buffered mode: large windows / no sink).
            sink.0.write_all(buf)?;
            buf.clear();
            consumed += n;
            total += produced;
        }
        Ok(total)
    })
}

/// Compress data into a zstd frame using our native encoder.
pub fn compress<W: Write + ?Sized, R: Read>(
    mut input: R,
    output: &mut W,
    level: Option<i32>,
    input_size: Option<usize>,
) -> io::Result<usize> {
    let level = api_level(level.unwrap_or(DEFAULT_COMPRESSION_LEVEL));
    crate::scratch_pair_with(|data, out| {
        if let Some(hint) = input_size {
            data.reserve(hint);
        }
        input.read_to_end(data)?;
        encode::encode_frame_into(out, data, level);
        output.write_all(out)?;
        Ok(out.len())
    })
}

/// Compress a byte slice into a new `Vec` (no intermediate input copy).
pub fn compress_bytes(input: &[u8], level: Option<i32>) -> Vec<u8> {
    let level = api_level(level.unwrap_or(DEFAULT_COMPRESSION_LEVEL));
    encode::encode_frame(input, level, Some(input.len() as u64))
}

/// Worst-case compressed size.
pub fn compress_bound(len: usize) -> usize {
    encode::compress_bound(len)
}

/// Streaming compressor with `ZSTD_compressStream2` semantics: `write`
/// feeds data (full 128 KiB blocks are emitted as they fill), `flush`
/// compresses everything pending so a decoder can consume all input written
/// so far (`ZSTD_e_flush`), `finish` writes the last block (`ZSTD_e_end`).
/// Generic over `W: Write` so callers can pass either a `Vec<u8>` or a
/// `Cursor<Vec<u8>>`.
pub struct ZstdStreamCompressor<W: Write = Vec<u8>> {
    enc: encode::StreamEncoder,
    out: Vec<u8>,
    output: W,
}

impl<W: Write> ZstdStreamCompressor<W> {
    pub fn new(output: W, level: i32) -> io::Result<Self> {
        Ok(Self {
            enc: encode::StreamEncoder::new(api_level(level)),
            out: Vec::new(),
            output,
        })
    }

    pub fn get_ref(&self) -> &W {
        &self.output
    }

    pub fn get_mut(&mut self) -> &mut W {
        &mut self.output
    }

    fn drain(&mut self) -> io::Result<()> {
        if !self.out.is_empty() {
            self.output.write_all(&self.out)?;
            self.out.clear();
        }
        Ok(())
    }

    pub fn finish(mut self) -> io::Result<W> {
        self.enc.end(&mut self.out);
        self.drain()?;
        Ok(self.output)
    }
}

impl<W: Write> Write for ZstdStreamCompressor<W> {
    fn write(&mut self, buf: &[u8]) -> io::Result<usize> {
        self.enc.push(&mut self.out, buf);
        self.drain()?;
        Ok(buf.len())
    }
    fn flush(&mut self) -> io::Result<()> {
        self.enc.flush(&mut self.out);
        self.drain()?;
        self.output.flush()
    }
}
