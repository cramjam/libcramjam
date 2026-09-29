//! bzip2 de/compression interface — C backend (libbzip2 via the `bzip2` crate).
use std::io::{self, Error, Read, Write};

use bzip2::read::{BzEncoder, MultiBzDecoder};

pub const BACKEND: crate::Backend = crate::Backend::C;

/// Default compression level, bzip2's default blockSize100k = 6.
pub const DEFAULT_COMPRESSION_LEVEL: u32 = 6;

/// libbzip2 only accepts block sizes 1..=9 (and panics in the crate
/// otherwise); clamp like the pure-Rust backend does.
fn compression(level: u32) -> bzip2::Compression {
    bzip2::Compression::new(level.clamp(1, 9))
}

/// Decompress via bzip2 (concatenated streams are decoded in sequence).
#[inline(always)]
pub fn decompress<W: Write + ?Sized, R: Read>(input: R, output: &mut W) -> Result<usize, Error> {
    let mut decoder = MultiBzDecoder::new(input);
    let n_bytes = io::copy(&mut decoder, output)?;
    Ok(n_bytes as usize)
}

/// Compress via bzip2.
#[inline(always)]
pub fn compress<W: Write + ?Sized, R: Read>(input: R, output: &mut W, level: Option<u32>) -> Result<usize, Error> {
    let level = level.unwrap_or(DEFAULT_COMPRESSION_LEVEL);
    let mut encoder = BzEncoder::new(input, compression(level));
    let n_bytes = io::copy(&mut encoder, output)?;
    Ok(n_bytes as usize)
}

/// Streaming compressor: `flush` makes everything written so far decodable
/// (`BZ_FLUSH`), `finish` ends the stream.
pub struct Bzip2StreamCompressor<W: Write>(bzip2::write::BzEncoder<W>);

impl<W: Write> Bzip2StreamCompressor<W> {
    pub fn new(output: W, level: u32) -> Self {
        Self(bzip2::write::BzEncoder::new(output, compression(level)))
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

impl<W: Write> Write for Bzip2StreamCompressor<W> {
    fn write(&mut self, buf: &[u8]) -> io::Result<usize> {
        self.0.write(buf)
    }
    fn flush(&mut self) -> io::Result<()> {
        self.0.flush()
    }
}
