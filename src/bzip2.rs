//! bzip2 de/compression interface — pure Rust implementation.
use std::io::{Error, Read, Write};

const DEFAULT_COMPRESSION_LEVEL: u32 = 6;

pub use crate::bzip2_impl::Bzip2StreamCompressor;

/// Decompress via bzip2.
#[inline(always)]
pub fn decompress<W: Write + ?Sized, R: Read>(mut input: R, output: &mut W) -> Result<usize, Error> {
    let mut data = Vec::new();
    input.read_to_end(&mut data)?;
    let mut decoded = Vec::new();
    crate::bzip2_impl::decode::decode_stream(&data, &mut decoded)?;
    output.write_all(&decoded)?;
    Ok(decoded.len())
}

/// Compress via bzip2.
#[inline(always)]
pub fn compress<W: Write + ?Sized, R: Read>(
    mut input: R,
    output: &mut W,
    level: Option<u32>,
) -> Result<usize, Error> {
    let level = level.unwrap_or(DEFAULT_COMPRESSION_LEVEL);
    let mut data = Vec::new();
    input.read_to_end(&mut data)?;
    let compressed = crate::bzip2_impl::encode::encode_stream(&data, level);
    output.write_all(&compressed)?;
    Ok(compressed.len())
}
