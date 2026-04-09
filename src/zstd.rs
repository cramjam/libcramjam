//! zstd de/compression interface
use std::io::{Error, Read, Write};

const DEFAULT_COMPRESSION_LEVEL: i32 = 0;

pub use crate::zstd_impl::ZstdStreamCompressor;

/// Get the max compressed length for a single pass
pub fn compress_bound(len: usize) -> usize {
    crate::zstd_impl::compress_bound(len)
}

/// Decompress zstd data
#[inline(always)]
pub fn decompress<W: Write + ?Sized, R: Read>(input: R, output: &mut W) -> Result<usize, Error> {
    crate::zstd_impl::decompress(input, output)
}

/// Compress zstd data
#[inline(always)]
pub fn compress<W: Write + ?Sized, R: Read>(
    input: R,
    output: &mut W,
    level: Option<i32>,
    input_size: Option<usize>,
) -> Result<usize, Error> {
    crate::zstd_impl::compress(input, output, level, input_size)
}
