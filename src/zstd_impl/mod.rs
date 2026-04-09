//! Pure-Rust Zstandard implementation (RFC 8878).
//!
//! Decompression: our own decoder (`decode::decode_frame`).
//! Compression: our own native encoder (`encode::encode_frame`) — raw blocks
//! at level 0, LZ77 + predefined-FSE sequences at level >= 1.

mod bits;
mod decode;
pub mod encode;
mod fse;
mod huf;

use std::io::{self, Read, Write};

/// Default zstd compression level — `1` means "use the entropy-coded path".
/// Level 0 is reserved for raw-block (store-only) output.
pub const DEFAULT_COMPRESSION_LEVEL: i32 = 1;

/// Decompress a zstd frame.
pub fn decompress<W: Write + ?Sized, R: Read>(
    mut input: R,
    output: &mut W,
) -> io::Result<usize> {
    let mut data = Vec::new();
    input.read_to_end(&mut data)?;

    let mut decoded = Vec::new();
    let mut consumed = 0usize;
    while consumed < data.len() {
        let n = decode::decode_frame(&data[consumed..], &mut decoded)?;
        if n == 0 {
            break;
        }
        consumed += n;
    }
    output.write_all(&decoded)?;
    Ok(decoded.len())
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
