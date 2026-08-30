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
    /// Build options from a preset (0..=9, optionally `| 0x8000_0000` for
    /// "extreme"). Exactly liblzma's `lzma_lzma_preset` table.
    pub fn new_preset(preset: u32) -> io::Result<Self> {
        const LEVEL_MASK: u32 = 0x1F;
        const EXTREME: u32 = 0x8000_0000;
        let level = preset & LEVEL_MASK;
        let flags = preset & !LEVEL_MASK;
        if level > 9 || (flags & !EXTREME) != 0 {
            return Err(io::Error::new(io::ErrorKind::InvalidInput, format!("xz: invalid preset {preset:#x}")));
        }
        const DICT_POW2: [u32; 10] = [18, 20, 21, 22, 22, 23, 23, 24, 25, 26];
        let dict_size = 1u32 << DICT_POW2[level as usize];

        let (mut mode, mut mf, mut nice_len, mut depth) = if level <= 3 {
            const DEPTHS: [u32; 4] = [4, 8, 24, 48];
            (
                Mode::Fast,
                if level == 0 { MatchFinder::HashChain3 } else { MatchFinder::HashChain4 },
                if level <= 1 { 128 } else { 273 },
                DEPTHS[level as usize],
            )
        } else {
            (
                Mode::Normal,
                MatchFinder::BinaryTree4,
                match level {
                    4 => 16,
                    5 => 32,
                    _ => 64,
                },
                0,
            )
        };
        if flags & EXTREME != 0 {
            mode = Mode::Normal;
            mf = MatchFinder::BinaryTree4;
            if level == 3 || level == 5 {
                nice_len = 192;
                depth = 0;
            } else {
                nice_len = 273;
                depth = 512;
            }
        }

        Ok(LzmaOptions {
            preset,
            dict_size,
            lc: 3,
            lp: 0,
            pb: 2,
            mode,
            nice_len,
            mf,
            depth,
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
