//! Zlib w/ ISA-L de/compression interface
#[cfg(all(
    any(feature = "izlib", feature = "izlib-static", feature = "izlib-shared"),
    target_pointer_width = "64",
))]
pub use isal;
use std::io::prelude::*;
use std::io::Error;

const DEFAULT_COMPRESSION_LEVEL: u32 = 3;

/// Decompress igzip data
#[inline(always)]
pub fn decompress<W: Write + ?Sized, R: Read>(input: R, output: &mut W) -> Result<usize, Error> {
    let mut decoder = isal::read::ZlibDecoder::new(input);
    let nbytes = std::io::copy(&mut decoder, output)?;
    Ok(nbytes as usize)
}

/// Compress igzip data
#[inline(always)]
pub fn compress<W: Write + ?Sized, R: Read>(
    input: R,
    output: &mut W,
    level: Option<u32>,
) -> Result<usize, Error> {
    let level = level.unwrap_or_else(|| DEFAULT_COMPRESSION_LEVEL);
    let level = isal::CompressionLevel::try_from(level as isize)?;
    let mut encoder = isal::read::ZlibEncoder::new(input, level);
    let n_bytes = std::io::copy(&mut encoder, output)?;
    Ok(n_bytes as usize)
}
