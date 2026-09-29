//! zlib de/compression interface — C backend (`flate2`, as in libcramjam 0.8).
use flate2::Compression;
use std::io::{self, Error, Read, Write};

pub const BACKEND: crate::Backend = crate::Backend::C;

pub const DEFAULT_COMPRESSION_LEVEL: u32 = 6;
const ZLIB_MIN_OVERHEAD: usize = 2 + 4; // header + footer

/// Compression upper bound
// xref: https://github.com/ebiggers/libdeflate/blob/6bb493615b0ef35c98fc4aa4ec04f448788db6a5/lib/zlib_compress.c#L77
pub fn compress_bound(input_len: usize) -> usize {
    ZLIB_MIN_OVERHEAD + crate::deflate::compress_bound(input_len)
}

/// Decompress zlib data
#[inline(always)]
pub fn decompress<W: Write + ?Sized, R: Read>(input: R, output: &mut W) -> Result<usize, Error> {
    crate::deflate::inflate(input, output, true)
}

/// Compress zlib data
#[inline(always)]
pub fn compress<W: Write + ?Sized, R: Read>(
    input: R,
    output: &mut W,
    level: Option<u32>,
) -> Result<usize, Error> {
    let level = level.unwrap_or(DEFAULT_COMPRESSION_LEVEL);
    let mut encoder = flate2::read::ZlibEncoder::new(input, Compression::new(level));
    let n_bytes = io::copy(&mut encoder, output)?;
    Ok(n_bytes as usize)
}

/// Streaming compressor: `flush` makes everything written so far decodable
/// (`Z_SYNC_FLUSH`), `finish` ends the stream.
pub struct ZlibStreamCompressor<W: Write>(flate2::write::ZlibEncoder<W>);

impl<W: Write> ZlibStreamCompressor<W> {
    pub fn new(output: W, level: u32) -> Self {
        Self(flate2::write::ZlibEncoder::new(output, Compression::new(level)))
    }
    pub fn get_ref(&self) -> &W {
        self.0.get_ref()
    }
    pub fn get_mut(&mut self) -> &mut W {
        self.0.get_mut()
    }
    pub fn finish(self) -> io::Result<W> {
        self.0.finish()
    }
}

impl<W: Write> Write for ZlibStreamCompressor<W> {
    fn write(&mut self, buf: &[u8]) -> io::Result<usize> {
        self.0.write(buf)
    }
    fn flush(&mut self) -> io::Result<()> {
        self.0.flush()
    }
}
