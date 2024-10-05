//! zlib de/compression interface
pub use flate2;
use flate2::read::{ZlibDecoder, ZlibEncoder};
use flate2::Compression;
use std::io::prelude::*;
use std::io::{Cursor, Error};

const DEFAULT_COMPRESSION_LEVEL: u32 = 6;

/// Decompress zlib data
#[inline(always)]
pub fn decompress<W: Write + ?Sized, R: Read>(input: R, output: &mut W) -> Result<usize, Error> {
    let mut decoder = ZlibDecoder::new(input);
    let mut out = vec![];
    let n_bytes = decoder.read_to_end(&mut out)?;
    std::io::copy(&mut Cursor::new(out.as_slice()), output)?;
    Ok(n_bytes as usize)
}

/// Compress zlib data
#[inline(always)]
pub fn compress<W: Write + ?Sized, R: Read>(
    input: R,
    output: &mut W,
    level: Option<u32>,
) -> Result<usize, Error> {
    let level = level.unwrap_or_else(|| DEFAULT_COMPRESSION_LEVEL);
    let mut encoder = ZlibEncoder::new(input, Compression::new(level));
    let n_bytes = std::io::copy(&mut encoder, output)?;
    Ok(n_bytes as usize)
}
