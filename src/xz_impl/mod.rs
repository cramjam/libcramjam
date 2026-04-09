//! Pure-Rust XZ / LZMA / LZMA2 implementation.
//!
//! Layered structure (mirrors liblzma):
//!
//! ```text
//! ┌────────────────────────────────────────────────────────────┐
//! │ xz_format     — .xz stream framing (header / blocks /      │
//! │                  index / footer / CRC32 / CRC64)            │
//! ├────────────────────────────────────────────────────────────┤
//! │ lzma2         — LZMA2 chunked layer (control byte +        │
//! │                  per-chunk dict reset rules)                │
//! ├────────────────────────────────────────────────────────────┤
//! │ lzma          — LZMA stream coder (literals, lengths,      │
//! │                  distances, repeat-distance state machine) │
//! ├────────────────────────────────────────────────────────────┤
//! │ range_coder   — bit-level range coder (encode + decode)    │
//! ├────────────────────────────────────────────────────────────┤
//! │ check         — CRC32 / CRC64 / SHA256 integrity checks    │
//! │ bcj           — Branch/Call/Jump filters (x86, arm, etc.)  │
//! └────────────────────────────────────────────────────────────┘
//! ```
//!
//! References:
//!   - The .xz file format specification: <https://tukaani.org/xz/xz-file-format.txt>
//!   - liblzma source under `lzma-sys/xz-5.2/src/liblzma/`
//!
//! All numeric constants and probability-update rules are taken straight
//! from liblzma's `range_common.h` / `lzma_common.h`.

pub mod alone;
pub mod bcj;
pub mod check;
pub mod options;
pub mod range_coder;
pub mod lzma;
pub mod lzma2;
pub mod xz_format;

pub use options::{Check, Filter, Filters, Format, LzmaOptions, MatchFinder, Mode};

use std::io::{self, Read, Write};

/// One-shot encode of `input` into a `.xz` stream at the given preset (0..=9).
pub fn encode_xz(input: &[u8], preset: u32) -> io::Result<Vec<u8>> {
    let mut out = Vec::with_capacity(input.len() / 2);
    xz_format::encode_xz_stream(input, preset, Check::Crc64, &mut out)?;
    Ok(out)
}

/// One-shot decode of a compressed stream into a Vec.  Auto-detects
/// `.xz` (the modern framed format, magic `FD 37 7A 58 5A 00`) vs the
/// legacy `.lzma` "Alone" format (13-byte header followed by a raw LZMA
/// stream — what `lzma.compress(..., format=FORMAT_ALONE)` produces).
pub fn decode_xz(input: &[u8]) -> io::Result<Vec<u8>> {
    let mut out = Vec::with_capacity(input.len() * 4);
    if alone::looks_like_alone(input) {
        alone::decode_alone(input, &mut out)?;
    } else {
        xz_format::decode_xz_stream(input, &mut out)?;
    }
    Ok(out)
}

/// Streaming Write-adapter so the cramjam Python wrapper can wrap an output
/// sink the same way it currently does with `xz2::write::XzEncoder`.
/// Buffers all input then encodes once on `finish` (matching the bzip2/zstd
/// adapters in this crate).
pub struct XzStreamCompressor<W: Write> {
    input: Vec<u8>,
    output: W,
    preset: u32,
}

impl<W: Write> XzStreamCompressor<W> {
    pub fn new(output: W, preset: u32) -> Self {
        Self {
            input: Vec::new(),
            output,
            preset,
        }
    }

    pub fn get_ref(&self) -> &W {
        &self.output
    }

    /// Encode the buffered input and write it to the underlying sink, then
    /// return the sink.
    pub fn finish(mut self) -> io::Result<W> {
        let compressed = encode_xz(&self.input, self.preset)?;
        self.output.write_all(&compressed)?;
        Ok(self.output)
    }
}

impl<W: Write> Write for XzStreamCompressor<W> {
    fn write(&mut self, buf: &[u8]) -> io::Result<usize> {
        self.input.extend_from_slice(buf);
        Ok(buf.len())
    }
    fn flush(&mut self) -> io::Result<()> {
        Ok(())
    }
}

/// Streaming Read-adapter mirror of the encoder side, used by the same
/// Python wrapper plumbing.
pub struct XzStreamDecompressor<R: Read> {
    /// Decoded bytes, ready to be served to the consumer.
    decoded: Vec<u8>,
    pos: usize,
    /// Source we still need to drain on the first read.
    source: Option<R>,
}

impl<R: Read> XzStreamDecompressor<R> {
    pub fn new(source: R) -> Self {
        Self {
            decoded: Vec::new(),
            pos: 0,
            source: Some(source),
        }
    }
}

impl<R: Read> Read for XzStreamDecompressor<R> {
    fn read(&mut self, buf: &mut [u8]) -> io::Result<usize> {
        if let Some(mut src) = self.source.take() {
            let mut all = Vec::new();
            src.read_to_end(&mut all)?;
            self.decoded = decode_xz(&all)?;
            self.pos = 0;
        }
        let remaining = self.decoded.len() - self.pos;
        let n = remaining.min(buf.len());
        buf[..n].copy_from_slice(&self.decoded[self.pos..self.pos + n]);
        self.pos += n;
        Ok(n)
    }
}
