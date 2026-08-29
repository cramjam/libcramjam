//! Pure-Rust Zstandard implementation (RFC 8878).
//!
//! Decompression: our own decoder (`decode::decode_frame`).
//! Compression: our own native encoder (`encode::encode_frame`) — raw blocks
//! at level 0, LZ77 + predefined-FSE sequences at level >= 1.

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
/// Level 0 emits raw blocks (store-only); levels >= 1 use entropy coding.
pub const DEFAULT_COMPRESSION_LEVEL: i32 = 3;

/// Decompress a zstd frame.
pub fn decompress<W: Write + ?Sized, R: Read>(
    mut input: R,
    output: &mut W,
) -> io::Result<usize> {
    let mut data = Vec::new();
    input.read_to_end(&mut data)?;

    let mut sink = crate::SinkRef(output);
    crate::scratch_with(|buf| {
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
    let level = level.unwrap_or(DEFAULT_COMPRESSION_LEVEL);
    let mut data = Vec::new();
    if let Some(hint) = input_size {
        data.reserve(hint);
    }
    input.read_to_end(&mut data)?;

    let compressed = encode::encode_frame(&data, level, Some(data.len() as u64));
    output.write_all(&compressed)?;
    Ok(compressed.len())
}

/// Compress a byte slice into a new `Vec` (no intermediate input copy).
pub fn compress_bytes(input: &[u8], level: Option<i32>) -> Vec<u8> {
    let level = level.unwrap_or(DEFAULT_COMPRESSION_LEVEL);
    encode::encode_frame(input, level, Some(input.len() as u64))
}

/// Worst-case compressed size.
pub fn compress_bound(len: usize) -> usize {
    encode::compress_bound(len)
}

/// Streaming compressor (Write-adapter for the C API).  Generic over
/// `W: Write` so callers can pass either a `Vec<u8>` or `Cursor<Vec<u8>>`.
pub struct ZstdStreamCompressor<W: Write = Vec<u8>> {
    input: Vec<u8>,
    output: W,
    level: i32,
}

impl<W: Write> ZstdStreamCompressor<W> {
    pub fn new(output: W, level: i32) -> io::Result<Self> {
        Ok(Self {
            input: Vec::new(),
            output,
            level,
        })
    }

    pub fn get_ref(&self) -> &W {
        &self.output
    }

    pub fn finish(mut self) -> io::Result<W> {
        let mut buf = Vec::with_capacity(self.input.len() / 2);
        compress(
            &mut std::io::Cursor::new(self.input),
            &mut buf,
            Some(self.level),
            None,
        )?;
        self.output.write_all(&buf)?;
        Ok(self.output)
    }
}

impl<W: Write> Write for ZstdStreamCompressor<W> {
    fn write(&mut self, buf: &[u8]) -> io::Result<usize> {
        self.input.extend_from_slice(buf);
        Ok(buf.len())
    }

    fn flush(&mut self) -> io::Result<()> {
        Ok(())
    }
}
