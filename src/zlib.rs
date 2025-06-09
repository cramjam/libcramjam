//! zlib de/compression interface
pub use flate2;
use flate2::read::{ZlibDecoder, ZlibEncoder};
use flate2::Compression;
use std::io::prelude::*;
use std::io::{Cursor, Error};

const DEFAULT_COMPRESSION_LEVEL: u32 = 6;

pub const ZLIB_MIN_HEADER_SIZE: usize = 2;
pub const ZLIB_FOOTER_SIZE: usize = 4;
pub const ZLIB_MIN_OVERHEAD: usize = ZLIB_MIN_HEADER_SIZE + ZLIB_FOOTER_SIZE;

/// Compression upper bound
// xref: https://github.com/ebiggers/libdeflate/blob/6bb493615b0ef35c98fc4aa4ec04f448788db6a5/lib/zlib_compress.c#L77
pub fn compress_bound(len: usize) -> usize {
    ZLIB_MIN_OVERHEAD + crate::deflate::compress_bound(len)
}

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
