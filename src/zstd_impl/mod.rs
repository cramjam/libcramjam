//! Pure-Rust Zstandard implementation (RFC 8878).
//!
//! Decompression uses `ruzstd` (proven, RFC-compliant).
//! Compression uses our own frame encoder (raw blocks for now).

mod bits;
mod decode;
pub mod encode;
mod fse;
mod huf;

use std::io::{self, Read, Write};

pub const DEFAULT_COMPRESSION_LEVEL: i32 = 0;

/// Decompress a zstd frame.
pub fn decompress<W: Write + ?Sized, R: Read>(
    input: R,
    output: &mut W,
) -> io::Result<usize> {
    let mut decoder = ruzstd::decoding::StreamingDecoder::new(input)
        .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e.to_string()))?;
    let n = io::copy(&mut decoder, output)?;
    Ok(n as usize)
}

/// Compress data into a zstd frame.
pub fn compress<W: Write + ?Sized, R: Read>(
    mut input: R,
    output: &mut W,
    level: Option<i32>,
    _input_size: Option<usize>,
) -> io::Result<usize> {
    let level = level.unwrap_or(DEFAULT_COMPRESSION_LEVEL);
    let mut data = Vec::new();
    input.read_to_end(&mut data)?;

    // ruzstd supports Fastest (~level 1) and Default (~level 3).
    // Use Fastest for now as it's the most stable.
    let clevel = ruzstd::encoding::CompressionLevel::Fastest;
    let compressed = ruzstd::encoding::compress_to_vec(data.as_slice(), clevel);

    output.write_all(&compressed)?;
    Ok(compressed.len())
}

/// Worst-case compressed size.
pub fn compress_bound(len: usize) -> usize {
    encode::compress_bound(len)
}

/// Streaming compressor (Write-adapter for the C API).
pub struct ZstdStreamCompressor {
    input: Vec<u8>,
    output: Vec<u8>,
    level: i32,
}

impl ZstdStreamCompressor {
    pub fn new(output: Vec<u8>, level: i32) -> io::Result<Self> {
        Ok(Self {
            input: Vec::new(),
            output,
            level,
        })
    }

    pub fn get_ref(&self) -> &Vec<u8> {
        &self.output
    }

    pub fn finish(self) -> io::Result<Vec<u8>> {
        let mut output = self.output;
        compress(
            &mut std::io::Cursor::new(self.input),
            &mut output,
            Some(self.level),
            None,
        )?;
        Ok(output)
    }
}

impl Write for ZstdStreamCompressor {
    fn write(&mut self, buf: &[u8]) -> io::Result<usize> {
        self.input.extend_from_slice(buf);
        Ok(buf.len())
    }

    fn flush(&mut self) -> io::Result<()> {
        Ok(())
    }
}
