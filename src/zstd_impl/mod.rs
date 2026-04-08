//! Pure-Rust Zstandard implementation (RFC 8878).
//!
//! Decompression: our own decoder (`decode::decode_frame`).
//! Compression: our own raw-block encoder at level 0; `ruzstd` for level ≥ 1
//! pending a native compressed-block encoder.

mod bits;
mod decode;
pub mod encode;
mod fse;
mod huf;

use std::io::{self, Read, Write};

pub const DEFAULT_COMPRESSION_LEVEL: i32 = 0;

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

/// Compress data into a zstd frame.
///
/// TODO: replace `ruzstd::encoding` with our own LZ77+FSE+Huffman encoder.
/// The native raw-block encoder in `encode::encode_frame` is intentionally
/// NOT wired here because it produces output larger than input for level 0.
pub fn compress<W: Write + ?Sized, R: Read>(
    mut input: R,
    output: &mut W,
    level: Option<i32>,
    _input_size: Option<usize>,
) -> io::Result<usize> {
    let _ = level.unwrap_or(DEFAULT_COMPRESSION_LEVEL);
    let mut data = Vec::new();
    input.read_to_end(&mut data)?;

    // ruzstd::encoding only ships Fastest and Uncompressed; the higher
    // levels are UNIMPLEMENTED in upstream.  Always use Fastest until we
    // have our own encoder.
    let compressed = ruzstd::encoding::compress_to_vec(
        data.as_slice(),
        ruzstd::encoding::CompressionLevel::Fastest,
    );

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
