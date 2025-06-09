//! deflate de/compression interface
pub use flate2;
use flate2::read::{DeflateDecoder, DeflateEncoder};
use flate2::Compression;
use std::io::prelude::*;
use std::io::Error;

pub const DEFAULT_COMPRESSION_LEVEL: u32 = 6;
pub const MIN_BLOCK_LENGTH: usize = 5_000;

/// Compression upper bound
// xref: https://github.com/ebiggers/libdeflate/blob/6bb493615b0ef35c98fc4aa4ec04f448788db6a5/lib/deflate_compress.c#L4081
pub fn compress_bound(input_len: usize) -> usize {
    let max_blocks = std::cmp::max((input_len + MIN_BLOCK_LENGTH - 1) / MIN_BLOCK_LENGTH, 1);
    (5 * max_blocks) + input_len
}

/// Decompress gzip data
#[inline(always)]
pub fn decompress<W: Write + ?Sized, R: Read>(input: R, output: &mut W) -> Result<usize, Error> {
    let mut decoder = DeflateDecoder::new(input);
    let n_bytes = std::io::copy(&mut decoder, output)?;
    Ok(n_bytes as usize)
}

/// Compress gzip data
#[inline(always)]
pub fn compress<W: Write + ?Sized, R: Read>(
    input: R,
    output: &mut W,
    level: Option<u32>,
) -> Result<usize, Error> {
    let level = level.unwrap_or_else(|| DEFAULT_COMPRESSION_LEVEL);

    let mut encoder = DeflateEncoder::new(input, Compression::new(level));
    let n_bytes = std::io::copy(&mut encoder, output)?;
    Ok(n_bytes as usize)
}
