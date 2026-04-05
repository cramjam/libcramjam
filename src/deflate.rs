//! deflate de/compression interface
use std::io::prelude::*;
use std::io::Error;

pub const DEFAULT_COMPRESSION_LEVEL: u32 = 6;
pub const MIN_BLOCK_LENGTH: usize = 5_000;

/// Compression upper bound
pub fn compress_bound(input_len: usize) -> usize {
    crate::deflate_impl::deflate_compress_bound(input_len)
}

/// Decompress deflate data
#[inline(always)]
pub fn decompress<W: Write + ?Sized, R: Read>(input: R, output: &mut W) -> Result<usize, Error> {
    crate::deflate_impl::deflate_decompress(input, output)
}

/// Compress deflate data
#[inline(always)]
pub fn compress<W: Write + ?Sized, R: Read>(
    input: R,
    output: &mut W,
    level: Option<u32>,
) -> Result<usize, Error> {
    crate::deflate_impl::deflate_compress(input, output, level)
}
