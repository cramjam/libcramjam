//! zlib de/compression interface
use std::io::prelude::*;
use std::io::Error;

pub const BACKEND: crate::Backend = crate::Backend::PureRust;

pub const DEFAULT_COMPRESSION_LEVEL: u32 = 6;

pub use crate::deflate_impl::ZlibStreamCompressor;

/// Compression upper bound
pub fn compress_bound(len: usize) -> usize {
    crate::deflate_impl::zlib_compress_bound(len)
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
