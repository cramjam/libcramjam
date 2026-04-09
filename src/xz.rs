//! XZ / LZMA / LZMA-Alone de/compression interface — pure-Rust backend.
//!
//! All public types and the function shapes match what the Python `cramjam`
//! crate expects from `libcramjam::xz::*` so the wrapper compiles unchanged
//! after the migration.  The internals call into `crate::xz_impl` instead
//! of `xz2`.
use std::io::{self, Read, Result, Write};

// Re-export the pure-Rust API types so cramjam-python's `libcramjam::xz::Format`
// (etc.) imports keep working.
pub use crate::xz_impl::options::{
    Check, Filter, Filters, Format, LzmaOptions, MatchFinder, Mode,
};
pub use crate::xz_impl::{XzStreamCompressor, XzStreamDecompressor};

/// Compatibility shim: the cramjam Python wrapper currently imports its
/// streaming encoder type as `libcramjam::xz::write::XzEncoder`.  We map
/// that to the pure-Rust `XzStreamCompressor` so the wrapper compiles
/// after the migration without renaming the field.
pub mod write {
    pub use crate::xz_impl::XzStreamCompressor as XzEncoder;
}

/// Same compatibility shim for the read side.  `XzDecoder` is a
/// `Read`-shaped wrapper that drains its source and decodes once on the
/// first read.
pub mod read {
    pub use crate::xz_impl::XzStreamDecompressor as XzDecoder;
}

const DEFAULT_PRESET: u32 = 6;

/// Decompress an XZ / LZMA stream from `input` into `output`.
#[inline(always)]
pub fn decompress<W: Write + ?Sized, R: Read>(mut input: R, output: &mut W) -> Result<usize> {
    let mut data = Vec::new();
    input.read_to_end(&mut data)?;
    let decoded = crate::xz_impl::decode_xz(&data)?;
    output.write_all(&decoded)?;
    Ok(decoded.len())
}

/// Compress an input stream as `.xz`.
///
/// `preset` is the LZMA compression preset (0..=9, default 6).
/// `format`, `check`, `filters` and `options` are accepted for API
/// compatibility with the previous xz2-backed version; the current
/// pure-Rust encoder honours `preset` and `check` and uses LZMA2 with
/// the preset's default options.  Filter chains and custom `LzmaOptions`
/// fall back to the preset for now (a TODO covered by the migration).
#[inline(always)]
pub fn compress<W: Write + ?Sized, R: Read>(
    mut input: R,
    output: &mut W,
    preset: Option<u32>,
    format: Option<impl Into<Format>>,
    check: Option<impl Into<Check>>,
    _filters: Option<impl Into<Filters>>,
    _options: Option<impl Into<LzmaOptions>>,
) -> Result<usize> {
    let preset = preset.unwrap_or(DEFAULT_PRESET);
    let format = format.map(Into::into).unwrap_or_default();
    let check = check.map(Into::into).unwrap_or_default();

    let mut data = Vec::new();
    input.read_to_end(&mut data)?;

    let compressed = match format {
        Format::AUTO | Format::XZ => {
            let mut out = Vec::with_capacity(data.len() / 2);
            crate::xz_impl::xz_format::encode_xz_stream(&data, preset, check, &mut out)?;
            out
        }
        Format::ALONE => {
            return Err(io::Error::new(
                io::ErrorKind::Unsupported,
                "xz: ALONE (.lzma) format encoder not yet implemented in the pure-Rust backend",
            ));
        }
        Format::RAW => {
            return Err(io::Error::new(
                io::ErrorKind::Unsupported,
                "xz: RAW filter-chain encoder not yet implemented in the pure-Rust backend",
            ));
        }
    };
    output.write_all(&compressed)?;
    Ok(compressed.len())
}
