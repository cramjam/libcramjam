//! Public configuration types for the XZ encoder.
//!
//! These mirror the names and shape of the equivalent types in liblzma /
//! the `xz2` crate so the cramjam Python wrapper can keep using the
//! `libcramjam::xz::*` import paths it has today.  All types are pure-Rust
//! enums and POD structs — no C bindings.

use std::io;

/// Top-level container format selector.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Format {
    /// Auto-select.  For compression this is `XZ`; for decompression the
    /// reader sniffs the magic bytes.
    AUTO,
    /// The modern `.xz` framed format (default).
    XZ,
    /// Legacy `.lzma` ("LZMA-Alone") format.
    ALONE,
    /// Raw stream — no framing, no integrity check, just LZMA bytes.
    RAW,
}

impl Default for Format {
    fn default() -> Self {
        Format::XZ
    }
}

/// Integrity check that the .xz container appends to each block.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Check {
    Crc64,
    Crc32,
    Sha256,
    None,
}

impl Default for Check {
    fn default() -> Self {
        Check::Crc64
    }
}

impl Check {
    /// Numeric "Check ID" stored in the .xz stream header (per the spec).
    pub(crate) fn id(self) -> u8 {
        match self {
            Check::None => 0x00,
            Check::Crc32 => 0x01,
            Check::Crc64 => 0x04,
            Check::Sha256 => 0x0A,
        }
    }

    pub(crate) fn from_id(id: u8) -> io::Result<Self> {
        match id {
            0x00 => Ok(Check::None),
            0x01 => Ok(Check::Crc32),
            0x04 => Ok(Check::Crc64),
            0x0A => Ok(Check::Sha256),
            _ => Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!("xz: unsupported check id 0x{:02x}", id),
            )),
        }
    }

    /// Number of bytes the check field occupies in a block footer.
    pub(crate) fn size(self) -> usize {
        match self {
            Check::None => 0,
            Check::Crc32 => 4,
            Check::Crc64 => 8,
            Check::Sha256 => 32,
        }
    }
}

/// Filter identifier.  These are the BCJ ("Branch/Call/Jump") and LZMA
/// filter types liblzma exposes.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Filter {
    /// LZMA2 — the default and only filter that exists at the end of every
    /// xz block's filter chain.
    Lzma2,
    /// Legacy LZMA1 (used in `.lzma` "Alone" format).
    Lzma1,
    Arm,
    ArmThumb,
    Ia64,
    PowerPC,
    Sparc,
    X86,
}

impl Default for Filter {
    fn default() -> Self {
        Filter::Lzma2
    }
}

/// LZMA match finder.  liblzma supports five variants; we expose the same
/// names and accept all of them, but several map to the same internal
/// implementation in this MVP — see the encoder for details.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MatchFinder {
    HashChain3,
    HashChain4,
    BinaryTree2,
    BinaryTree3,
    BinaryTree4,
}

/// Encoder mode.  `Fast` greedy parser, `Normal` near-optimal parser.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Mode {
    Fast,
    Normal,
}

/// LZMA filter options.  Equivalent to liblzma's `lzma_options_lzma`.
#[derive(Clone, Debug)]
pub struct LzmaOptions {
    pub preset: u32,
    pub dict_size: u32,
    pub lc: u32,
    pub lp: u32,
    pub pb: u32,
    pub mode: Mode,
    pub nice_len: u32,
    pub mf: MatchFinder,
    pub depth: u32,
}

impl Default for LzmaOptions {
    fn default() -> Self {
        Self::new_preset(6).expect("preset 6 is always valid")
    }
}

impl LzmaOptions {
    /// Build options from a preset (0..=9).  These match liblzma's preset
    /// table from `lzma_encoder_presets.c`.
    pub fn new_preset(preset: u32) -> io::Result<Self> {
        let p = preset & 0x07;
        let extreme = (preset & 0x80000000) != 0;

        // dict_size from liblzma's preset_dict_size table.
        let dict_size = match p {
            0 => 1u32 << 16,         // 64 KiB
            1 => 1u32 << 20,         // 1 MiB
            2 => 1u32 << 21,         // 2 MiB
            3 => 1u32 << 22,         // 4 MiB
            4 => 1u32 << 22,         // 4 MiB
            5 => 1u32 << 23,         // 8 MiB
            6 => 1u32 << 23,         // 8 MiB
            7 => 1u32 << 24,         // 16 MiB
            _ => 1u32 << 25,         // 32 MiB (presets 8 and 9)
        };

        // (mode, nice_len, mf, depth) from the same table.
        let (mode, nice_len, mf, depth_extra) = match p {
            0 => (Mode::Fast, 128u32, MatchFinder::HashChain4, 0u32),
            1 => (Mode::Fast, 128, MatchFinder::HashChain4, 0),
            2 => (Mode::Fast, 273, MatchFinder::HashChain4, 0),
            3 => (Mode::Normal, 32, MatchFinder::BinaryTree4, 0),
            4 => (Mode::Normal, 64, MatchFinder::BinaryTree4, 0),
            _ => (Mode::Normal, 64, MatchFinder::BinaryTree4, 0),
        };

        Ok(LzmaOptions {
            preset,
            dict_size,
            lc: 3,
            lp: 0,
            pb: 2,
            mode,
            nice_len: if extreme { 273 } else { nice_len },
            mf,
            depth: depth_extra,
        })
    }

    pub fn dict_size(&mut self, dict_size: u32) -> &mut Self {
        self.dict_size = dict_size;
        self
    }
    pub fn literal_context_bits(&mut self, lc: u32) -> &mut Self {
        self.lc = lc;
        self
    }
    pub fn literal_position_bits(&mut self, lp: u32) -> &mut Self {
        self.lp = lp;
        self
    }
    pub fn position_bits(&mut self, pb: u32) -> &mut Self {
        self.pb = pb;
        self
    }
    pub fn mode(&mut self, mode: Mode) -> &mut Self {
        self.mode = mode;
        self
    }
    pub fn nice_len(&mut self, nice_len: u32) -> &mut Self {
        self.nice_len = nice_len;
        self
    }
    pub fn match_finder(&mut self, mf: MatchFinder) -> &mut Self {
        self.mf = mf;
        self
    }
    pub fn depth(&mut self, depth: u32) -> &mut Self {
        self.depth = depth;
        self
    }
}

/// A filter chain.  In .xz a chain is at most 4 entries and the LAST entry
/// is always LZMA1/LZMA2; preceding entries are BCJ filters.
#[derive(Clone, Debug, Default)]
pub struct Filters {
    pub(crate) chain: Vec<FilterEntry>,
}

/// One entry in a filter chain.
#[derive(Clone, Debug)]
pub(crate) struct FilterEntry {
    pub filter: Filter,
    pub options: Option<LzmaOptions>,
}

impl Filters {
    pub fn new() -> Self {
        Self { chain: Vec::new() }
    }

    pub fn lzma1(&mut self, options: &LzmaOptions) -> &mut Self {
        self.chain.push(FilterEntry {
            filter: Filter::Lzma1,
            options: Some(options.clone()),
        });
        self
    }
    pub fn lzma2(&mut self, options: &LzmaOptions) -> &mut Self {
        self.chain.push(FilterEntry {
            filter: Filter::Lzma2,
            options: Some(options.clone()),
        });
        self
    }
    pub fn arm(&mut self) -> &mut Self {
        self.push_simple(Filter::Arm)
    }
    pub fn arm_thumb(&mut self) -> &mut Self {
        self.push_simple(Filter::ArmThumb)
    }
    pub fn ia64(&mut self) -> &mut Self {
        self.push_simple(Filter::Ia64)
    }
    pub fn powerpc(&mut self) -> &mut Self {
        self.push_simple(Filter::PowerPC)
    }
    pub fn sparc(&mut self) -> &mut Self {
        self.push_simple(Filter::Sparc)
    }
    pub fn x86(&mut self) -> &mut Self {
        self.push_simple(Filter::X86)
    }

    fn push_simple(&mut self, filter: Filter) -> &mut Self {
        self.chain.push(FilterEntry { filter, options: None });
        self
    }
}
