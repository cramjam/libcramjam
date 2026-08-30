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
pub mod lzma_enc;
pub mod lzma2;
pub mod raw;
pub mod xz_format;

pub use options::{Check, Filter, Filters, Format, LzmaOptions, MatchFinder, Mode};

/// Decode a raw (container-less, `Format::RAW`) stream produced with the given
/// filter chain — the chain must match the encoder's (liblzma's
/// `lzma_raw_decoder`).  LZMA entries without options default to preset 6.
pub fn decode_raw(input: &[u8], filters: &Filters, out: &mut Vec<u8>) -> io::Result<()> {
    let chain = filters.resolve(6, None)?;
    raw::decode_raw(input, &chain, out)
}

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
    let mut out = Vec::new();
    decode_xz_into(input, &mut out)?;
    Ok(out)
}

/// [`decode_xz`] into a caller-provided (possibly pre-allocated) buffer.
pub fn decode_xz_into(input: &[u8], out: &mut Vec<u8>) -> io::Result<()> {
    out.reserve(input.len().saturating_mul(4));
    if alone::looks_like_alone(input) {
        alone::decode_alone(input, out)?;
    } else {
        xz_format::decode_xz_stream(input, out)?;
    }
    Ok(())
}

/// Streaming `.xz` compressor with the semantics of liblzma's encoder
/// (`xz2::write::XzEncoder`): the stream and block headers are written up
/// front, `write` feeds the LZMA2 encoder (keeping only the dictionary
/// window plus a reserve in memory), `flush` is `LZMA_SYNC_FLUSH` — the
/// current chunk is finished so everything written so far decodes — and
/// `finish` is `LZMA_FINISH` (end marker, padding, check, index, footer).
pub struct XzStreamCompressor<W: Write> {
    output: W,
    inner: Option<XzStreamState>,
    /// Set when the preset was invalid; reported on first use.
    error: Option<String>,
}

struct XzStreamState {
    lzma2: lzma_enc::Lzma2StreamEncoder,
    check: Check,
    digest: XzDigest,
    dict_size: u32,
    started: bool,
    block_header_size: usize,
    payload_size: u64,
    total_in: u64,
}

enum XzDigest {
    None,
    Crc32(crc32fast::Hasher),
    Crc64(crc_fast::Digest),
}

impl XzDigest {
    fn new(check: Check) -> io::Result<Self> {
        Ok(match check {
            Check::None => Self::None,
            Check::Crc32 => Self::Crc32(crc32fast::Hasher::new()),
            Check::Crc64 => Self::Crc64(crc_fast::Digest::new(crc_fast::CrcAlgorithm::Crc64Xz)),
            Check::Sha256 => {
                return Err(io::Error::new(
                    io::ErrorKind::Unsupported,
                    "xz: SHA-256 check not yet supported",
                ))
            }
        })
    }
    fn update(&mut self, data: &[u8]) {
        match self {
            Self::None => {}
            Self::Crc32(h) => h.update(data),
            Self::Crc64(d) => d.update(data),
        }
    }
    fn finalize(self) -> Vec<u8> {
        match self {
            Self::None => Vec::new(),
            Self::Crc32(h) => h.finalize().to_le_bytes().to_vec(),
            Self::Crc64(d) => d.finalize().to_le_bytes().to_vec(),
        }
    }
}

impl<W: Write> XzStreamCompressor<W> {
    pub fn new(output: W, preset: u32) -> Self {
        match Self::state_for(preset) {
            Ok(inner) => Self { output, inner: Some(inner), error: None },
            Err(e) => Self { output, inner: None, error: Some(e.to_string()) },
        }
    }

    fn state_for(preset: u32) -> io::Result<XzStreamState> {
        let opts = LzmaOptions::new_preset(preset)?;
        let check = Check::Crc64;
        Ok(XzStreamState {
            lzma2: lzma_enc::Lzma2StreamEncoder::new(&opts)?,
            check,
            digest: XzDigest::new(check)?,
            dict_size: opts.dict_size,
            started: false,
            block_header_size: 0,
            payload_size: 0,
            total_in: 0,
        })
    }

    pub fn get_ref(&self) -> &W {
        &self.output
    }

    pub fn get_mut(&mut self) -> &mut W {
        &mut self.output
    }

    fn state(&mut self) -> io::Result<(&mut XzStreamState, &mut W)> {
        match (self.inner.as_mut(), &self.error) {
            (Some(st), _) => Ok((st, &mut self.output)),
            (None, Some(e)) => Err(io::Error::new(io::ErrorKind::InvalidInput, e.clone())),
            (None, None) => Err(io::Error::new(io::ErrorKind::Other, "xz: compressor already finished")),
        }
    }

    /// Stream header + block header, once.
    fn start(st: &mut XzStreamState, output: &mut W) -> io::Result<()> {
        if !st.started {
            let mut head = Vec::with_capacity(32);
            xz_format::write_stream_header(&mut head, st.check);
            st.block_header_size = xz_format::write_block_header(&mut head, &[], st.dict_size);
            output.write_all(&head)?;
            st.started = true;
        }
        Ok(())
    }

    fn drain(st: &mut XzStreamState, output: &mut W) -> io::Result<()> {
        if !st.lzma2.out.is_empty() {
            output.write_all(&st.lzma2.out)?;
            st.payload_size += st.lzma2.out.len() as u64;
            st.lzma2.out.clear();
        }
        Ok(())
    }

    /// `LZMA_FINISH`: close the stream and return the sink.
    pub fn finish(mut self) -> io::Result<W> {
        let (st, output) = self.state()?;
        Self::start(st, output)?;
        st.lzma2.finish_input(lzma_enc::Action::Finish);
        Self::drain(st, output)?;
        let mut st = self.inner.take().unwrap();
        let mut tail = Vec::with_capacity(64);
        let stream_flags = [0u8, st.check.id() & 0x0F];
        let digest = std::mem::replace(&mut st.digest, XzDigest::None);
        xz_format::write_block_trailer(
            &mut tail,
            st.block_header_size,
            st.payload_size,
            st.check,
            &digest.finalize(),
            st.total_in,
            stream_flags,
        );
        self.output.write_all(&tail)?;
        Ok(self.output)
    }
}

impl<W: Write> Write for XzStreamCompressor<W> {
    fn write(&mut self, buf: &[u8]) -> io::Result<usize> {
        let (st, output) = self.state()?;
        Self::start(st, output)?;
        st.digest.update(buf);
        st.total_in += buf.len() as u64;
        st.lzma2.write(buf);
        Self::drain(st, output)?;
        Ok(buf.len())
    }

    /// `LZMA_SYNC_FLUSH`: everything written so far becomes decodable.
    fn flush(&mut self) -> io::Result<()> {
        let (st, output) = self.state()?;
        Self::start(st, output)?;
        st.lzma2.finish_input(lzma_enc::Action::Flush);
        Self::drain(st, output)?;
        output.flush()
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
