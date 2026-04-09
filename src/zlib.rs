//! zlib de/compression interface
use std::io::prelude::*;
use std::io::Error;

const DEFAULT_COMPRESSION_LEVEL: u32 = 6;

pub const ZLIB_MIN_HEADER_SIZE: usize = 2;
pub const ZLIB_FOOTER_SIZE: usize = 4;
pub const ZLIB_MIN_OVERHEAD: usize = ZLIB_MIN_HEADER_SIZE + ZLIB_FOOTER_SIZE;

pub use crate::deflate_impl::ZlibStreamCompressor;

/// Compression upper bound
pub fn compress_bound(len: usize) -> usize {
    ZLIB_MIN_OVERHEAD + crate::deflate::compress_bound(len)
}

/// Decompress zlib data
#[inline(always)]
pub fn decompress<W: Write + ?Sized, R: Read>(input: R, output: &mut W) -> Result<usize, Error> {
    crate::deflate_impl::zlib_decompress(input, output)
}

/// Compress zlib data
#[inline(always)]
pub fn compress<W: Write + ?Sized, R: Read>(
    input: R,
    output: &mut W,
    level: Option<u32>,
) -> Result<usize, Error> {
    crate::deflate_impl::zlib_compress(input, output, level)
}
