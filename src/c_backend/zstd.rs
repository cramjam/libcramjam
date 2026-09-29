//! zstd de/compression interface — C backend (libzstd via the `zstd` crate).
use std::io::{self, Error, Read, Write};

pub const BACKEND: crate::Backend = crate::Backend::C;

/// Default compression level, `ZSTD_defaultCLevel()`.
pub const DEFAULT_COMPRESSION_LEVEL: i32 = 3;

/// Get the max compressed length for a single pass
pub fn compress_bound(len: usize) -> usize {
    zstd::zstd_safe::compress_bound(len)
}

/// Decompress zstd data
#[inline(always)]
pub fn decompress<W: Write + ?Sized, R: Read>(input: R, output: &mut W) -> Result<usize, Error> {
    let mut decoder = zstd::stream::read::Decoder::new(input)?;
    let n_bytes = io::copy(&mut decoder, output)?;
    Ok(n_bytes as usize)
}

/// Compress zstd data
#[inline(always)]
pub fn compress<W: Write + ?Sized, R: Read>(
    input: R,
    output: &mut W,
    level: Option<i32>,
    input_size: Option<usize>,
) -> Result<usize, Error> {
    let level = level.unwrap_or(DEFAULT_COMPRESSION_LEVEL);
    let mut encoder = zstd::stream::read::Encoder::new(input, level)?;
    encoder.set_pledged_src_size(input_size.map(|v| v as u64))?;
    let n_bytes = io::copy(&mut encoder, output)?;
    Ok(n_bytes as usize)
}

/// Streaming compressor: `write` feeds data, `flush` makes everything written
/// so far decodable, `finish` ends the frame.
pub struct ZstdStreamCompressor<W: Write>(zstd::stream::write::Encoder<'static, W>);

impl<W: Write> ZstdStreamCompressor<W> {
    pub fn new(output: W, level: i32) -> io::Result<Self> {
        zstd::stream::write::Encoder::new(output, level).map(Self)
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

impl<W: Write> Write for ZstdStreamCompressor<W> {
    fn write(&mut self, buf: &[u8]) -> io::Result<usize> {
        self.0.write(buf)
    }
    fn flush(&mut self) -> io::Result<()> {
        self.0.flush()
    }
}
