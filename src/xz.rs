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
use crate::xz_impl::options::ResolvedFilter;

/// Default compression preset, matching C xz's `LZMA_PRESET_DEFAULT` = 6.
pub const DEFAULT_COMPRESSION_LEVEL: u32 = 6;

/// Decompress an XZ / LZMA stream from `input` into `output`.
#[inline(always)]
pub fn decompress<W: Write + ?Sized, R: Read>(mut input: R, output: &mut W) -> Result<usize> {
    let mut data = Vec::new();
    input.read_to_end(&mut data)?;
    crate::with_scratch(output, |decoded| crate::xz_impl::decode_xz_into(&data, decoded))
}

/// Compress an input stream as `.xz`.
///
/// * `preset` — LZMA compression preset (0..=9, default 6).  Used as the
///   base for the encoder's `LzmaOptions` unless `options` overrides it.
/// * `format` — `XZ` (default), `AUTO` (same as XZ for compression),
///   `ALONE` (legacy `.lzma`: one LZMA1 stream, no filters), or `RAW`
///   (the bare filter-chain output, no container/check — decode with
///   [`decompress_raw`] and the same chain).
/// * `check` — block integrity check: CRC64 (default), CRC32, SHA256, None.
/// * `filters` — optional filter chain: up to three BCJ filters (x86, ARM,
///   ARM-Thumb, IA-64, PowerPC, SPARC) followed by LZMA2 (LZMA1 is only
///   valid in `RAW` / `ALONE`), exactly liblzma's rules.
/// * `options` — optional `LzmaOptions` override.  When provided,
///   replaces the per-preset defaults for lc/lp/pb/dict_size/etc.
#[inline(always)]
pub fn compress<W: Write + ?Sized, R: Read>(
    mut input: R,
    output: &mut W,
    preset: Option<u32>,
    format: Option<impl Into<Format>>,
    check: Option<impl Into<Check>>,
    filters: Option<impl Into<Filters>>,
    options: Option<impl Into<LzmaOptions>>,
) -> Result<usize> {
    let preset = preset.unwrap_or(DEFAULT_COMPRESSION_LEVEL);
    let format = format.map(Into::into).unwrap_or_default();
    let check = check.map(Into::into).unwrap_or_default();

    // Resolve + validate the filter chain (empty = a single LZMA2 filter).
    // LZMA options precedence: explicit `options` > the chain entry's
    // options > preset default.
    let options: Option<LzmaOptions> = options.map(Into::into);
    let chain = filters
        .map(Into::into)
        .unwrap_or_default()
        .resolve(preset, options.as_ref())?;

    let mut data = Vec::new();
    input.read_to_end(&mut data)?;

    let mut out = Vec::with_capacity(data.len() / 2 + 64);
    match format {
        Format::AUTO | Format::XZ => {
            crate::xz_impl::xz_format::encode_xz_stream_chain(&data, &chain, check, &mut out)?;
        }
        Format::ALONE => {
            // The .lzma container carries exactly one LZMA1 stream and no
            // filter flags; take the LZMA options from the chain's tail.
            let lzma_options = match chain.as_slice() {
                [ResolvedFilter::Lzma1(o)] | [ResolvedFilter::Lzma2(o)] => o,
                _ => {
                    return Err(io::Error::new(
                        io::ErrorKind::InvalidInput,
                        "xz: the .lzma (ALONE) format cannot carry BCJ filters",
                    ))
                }
            };
            crate::xz_impl::lzma_enc::encode_lzma_alone(&data, lzma_options, &mut out)?;
        }
        Format::RAW => {
            crate::xz_impl::raw::encode_raw(&data, &chain, &mut out)?;
        }
    }
    output.write_all(&out)?;
    Ok(out.len())
}

/// Decompress a raw (`Format::RAW`) stream: no container, so the filter chain
/// it was encoded with must be supplied (liblzma's `lzma_raw_decoder`).
pub fn decompress_raw<W: Write + ?Sized, R: Read>(
    mut input: R,
    output: &mut W,
    filters: impl Into<Filters>,
) -> Result<usize> {
    let mut data = Vec::new();
    input.read_to_end(&mut data)?;
    let filters = filters.into();
    crate::with_scratch(output, |decoded| crate::xz_impl::decode_raw(&data, &filters, decoded))
}
