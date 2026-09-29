//! gzip de/compression interface — C backend (`flate2`, as in libcramjam 0.8).
use flate2::Compression;
use std::io::{self, Error, Read, Write};

pub const BACKEND: crate::Backend = crate::Backend::C;

pub const DEFAULT_COMPRESSION_LEVEL: u32 = 6;
const GZIP_MIN_OVERHEAD: usize = 10 + 8; // header + footer

/// Compression upper bound
// xref: https://github.com/ebiggers/libdeflate/blob/6bb493615b0ef35c98fc4aa4ec04f448788db6a5/lib/gzip_compress.c#L85
pub fn compress_bound(input_len: usize) -> usize {
    GZIP_MIN_OVERHEAD + crate::deflate::compress_bound(input_len)
}

/// Decompress gzip data (concatenated members are decoded in sequence)
#[inline(always)]
pub fn decompress<W: Write + ?Sized, R: Read>(input: R, output: &mut W) -> Result<usize, Error> {
    let mut decoder = flate2::read::MultiGzDecoder::new(input);
    let n_bytes = io::copy(&mut decoder, output)?;
    Ok(n_bytes as usize)
}

/// Compress gzip data
#[inline(always)]
pub fn compress<W: Write + ?Sized, R: Read>(
    input: R,
    output: &mut W,
    level: Option<u32>,
) -> Result<usize, Error> {
    let level = level.unwrap_or(DEFAULT_COMPRESSION_LEVEL);
    let mut encoder = flate2::read::GzEncoder::new(input, Compression::new(level));
    let n_bytes = io::copy(&mut encoder, output)?;
    Ok(n_bytes as usize)
}

/// Streaming compressor: `flush` makes everything written so far decodable
/// (`Z_SYNC_FLUSH`), `finish` ends the stream.
pub struct GzipStreamCompressor<W: Write>(flate2::write::GzEncoder<W>);

impl<W: Write> GzipStreamCompressor<W> {
    pub fn new(output: W, level: u32) -> Self {
        Self(flate2::write::GzEncoder::new(output, Compression::new(level)))
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

impl<W: Write> Write for GzipStreamCompressor<W> {
    fn write(&mut self, buf: &[u8]) -> io::Result<usize> {
        self.0.write(buf)
    }
    fn flush(&mut self) -> io::Result<()> {
        self.0.flush()
    }
}
