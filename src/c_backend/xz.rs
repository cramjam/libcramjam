//! XZ / LZMA / LZMA-Alone de/compression interface — C backend (liblzma via
//! `xz2`, plus `lzma-sys` for the raw coder, which `xz2` doesn't wrap).
//!
//! The option types and filter-chain validation are shared with the
//! pure-Rust backend, so both accept and reject exactly the same inputs.
use std::io::{self, BufRead, BufReader, Read, Result, Write};

use xz2::read::{XzDecoder, XzEncoder};
use xz2::stream::Stream;

#[allow(dead_code)] // pure-backend helpers (check ids, sizes) live here too
#[path = "../xz_impl/options.rs"]
mod options;

use options::ResolvedFilter;
pub use options::{Check, Filter, Filters, Format, LzmaOptions, MatchFinder, Mode};

pub const BACKEND: crate::Backend = crate::Backend::C;

/// Default compression preset, matching C xz's `LZMA_PRESET_DEFAULT` = 6.
pub const DEFAULT_COMPRESSION_LEVEL: u32 = 6;

/// Decompress an XZ / LZMA stream from `input` into `output`.
/// Concatenated `.xz` streams (and stream padding) are decoded in sequence.
#[inline(always)]
pub fn decompress<W: Write + ?Sized, R: Read>(input: R, output: &mut W) -> Result<usize> {
    let xz_magicbytes = b"\xfd7zXZ\x00";
    let mut input = BufReader::new(input);
    let stream = {
        let innerbuf = input.fill_buf()?;
        if innerbuf.len() >= xz_magicbytes.len() && &innerbuf[..xz_magicbytes.len()] == xz_magicbytes {
            Stream::new_stream_decoder(u64::MAX, xz2::stream::TELL_ANY_CHECK | xz2::stream::CONCATENATED)?
        } else {
            Stream::new_lzma_decoder(u64::MAX)?
        }
    };
    let mut decoder = XzDecoder::new_stream(input, stream);
    let n_bytes = io::copy(&mut decoder, output)?;
    Ok(n_bytes as usize)
}

/// Compress an input stream; see the pure-Rust backend for the parameter
/// semantics, which are identical.
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
    let filters: Option<Filters> = filters.map(Into::into);
    let options: Option<LzmaOptions> = options.map(Into::into);
    let customized = filters.is_some() || options.is_some();
    let chain = filters.unwrap_or_default().resolve(preset, options.as_ref())?;

    let stream = match format {
        // Same bytes as a single-LZMA2 stream encoder; kept for exact 0.8 parity.
        Format::AUTO | Format::XZ if !customized => Stream::new_easy_encoder(preset, xz2_check(check))?,
        Format::AUTO | Format::XZ => Stream::new_stream_encoder(&xz2_filters(&chain)?, xz2_check(check))?,
        Format::ALONE => match chain.as_slice() {
            [ResolvedFilter::Lzma1(o)] | [ResolvedFilter::Lzma2(o)] => Stream::new_lzma_encoder(&xz2_lzma_options(o)?)?,
            _ => {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidInput,
                    "xz: the .lzma (ALONE) format cannot carry BCJ filters",
                ))
            }
        },
        Format::RAW => {
            let mut data = Vec::new();
            input.read_to_end(&mut data)?;
            let mut out = Vec::with_capacity(data.len() / 2 + 64);
            raw_code(&data, &chain, true, &mut out)?;
            output.write_all(&out)?;
            return Ok(out.len());
        }
    };
    let mut encoder = XzEncoder::new_stream(input, stream);
    let n_bytes = io::copy(&mut encoder, output)?;
    Ok(n_bytes as usize)
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
    let chain = filters.into().resolve(DEFAULT_COMPRESSION_LEVEL, None)?;
    let mut out = Vec::new();
    raw_code(&data, &chain, false, &mut out)?;
    output.write_all(&out)?;
    Ok(out.len())
}

/// Streaming `.xz` compressor (CRC64 check): `flush` is `LZMA_SYNC_FLUSH` —
/// everything written so far becomes decodable — and `finish` is
/// `LZMA_FINISH`. An invalid preset is reported on first use.
///
/// Drives liblzma directly rather than through `xz2::write::XzEncoder`,
/// whose `flush` (a) uses `LZMA_FULL_FLUSH` and (b) leaves the last flushed
/// chunk in its internal buffer until the next call, so a flushed prefix
/// didn't decode.
pub struct XzStreamCompressor<W: Write> {
    stream: io::Result<Stream>,
    output: W,
    buf: Vec<u8>,
}

impl<W: Write> XzStreamCompressor<W> {
    pub fn new(output: W, preset: u32) -> Self {
        let stream = LzmaOptions::new_preset(preset)
            .and_then(|_| Stream::new_easy_encoder(preset, xz2::stream::Check::Crc64).map_err(io::Error::from));
        Self { stream, output, buf: Vec::with_capacity(32 << 10) }
    }

    pub fn get_ref(&self) -> &W {
        &self.output
    }

    pub fn get_mut(&mut self) -> &mut W {
        &mut self.output
    }

    /// Feed `input` with `action`, writing all produced bytes to the sink.
    /// `Run` returns once the input is consumed; flush/finish run to
    /// `StreamEnd`.
    fn process(&mut self, mut input: &[u8], action: xz2::stream::Action) -> io::Result<()> {
        use xz2::stream::{Action, Status};
        let stream = match &mut self.stream {
            Ok(s) => s,
            Err(e) => return Err(io::Error::new(io::ErrorKind::InvalidInput, e.to_string())),
        };
        loop {
            self.buf.clear();
            let before = stream.total_in();
            let status = stream.process_vec(input, &mut self.buf, action)?;
            input = &input[(stream.total_in() - before) as usize..];
            self.output.write_all(&self.buf)?;
            let done = match action {
                Action::Run => input.is_empty(),
                _ => status == Status::StreamEnd,
            };
            if done {
                return Ok(());
            }
        }
    }

    pub fn finish(mut self) -> io::Result<W> {
        self.process(&[], xz2::stream::Action::Finish)?;
        Ok(self.output)
    }
}

impl<W: Write> Write for XzStreamCompressor<W> {
    fn write(&mut self, buf: &[u8]) -> io::Result<usize> {
        self.process(buf, xz2::stream::Action::Run)?;
        Ok(buf.len())
    }
    fn flush(&mut self) -> io::Result<()> {
        self.process(&[], xz2::stream::Action::SyncFlush)?;
        self.output.flush()
    }
}

/// Read adapter that decodes the whole source on first read.
pub struct XzStreamDecompressor<R: Read> {
    decoded: io::Cursor<Vec<u8>>,
    source: Option<R>,
}

impl<R: Read> XzStreamDecompressor<R> {
    pub fn new(source: R) -> Self {
        Self { decoded: io::Cursor::new(Vec::new()), source: Some(source) }
    }
}

impl<R: Read> Read for XzStreamDecompressor<R> {
    fn read(&mut self, buf: &mut [u8]) -> io::Result<usize> {
        if let Some(src) = self.source.take() {
            decompress(src, self.decoded.get_mut())?;
        }
        self.decoded.read(buf)
    }
}

fn xz2_check(check: Check) -> xz2::stream::Check {
    match check {
        Check::Crc64 => xz2::stream::Check::Crc64,
        Check::Crc32 => xz2::stream::Check::Crc32,
        Check::Sha256 => xz2::stream::Check::Sha256,
        Check::None => xz2::stream::Check::None,
    }
}

fn xz2_lzma_options(o: &LzmaOptions) -> io::Result<xz2::stream::LzmaOptions> {
    let mut x = xz2::stream::LzmaOptions::new_preset(o.preset)?;
    x.dict_size(o.dict_size)
        .literal_context_bits(o.lc)
        .literal_position_bits(o.lp)
        .position_bits(o.pb)
        .nice_len(o.nice_len)
        .depth(o.depth)
        .mode(match o.mode {
            Mode::Fast => xz2::stream::Mode::Fast,
            Mode::Normal => xz2::stream::Mode::Normal,
        })
        .match_finder(match o.mf {
            MatchFinder::HashChain3 => xz2::stream::MatchFinder::HashChain3,
            MatchFinder::HashChain4 => xz2::stream::MatchFinder::HashChain4,
            MatchFinder::BinaryTree2 => xz2::stream::MatchFinder::BinaryTree2,
            MatchFinder::BinaryTree3 => xz2::stream::MatchFinder::BinaryTree3,
            MatchFinder::BinaryTree4 => xz2::stream::MatchFinder::BinaryTree4,
        });
    Ok(x)
}

fn xz2_filters(chain: &[ResolvedFilter]) -> io::Result<xz2::stream::Filters> {
    let mut f = xz2::stream::Filters::new();
    for entry in chain {
        match entry {
            ResolvedFilter::Bcj(options::FILTER_X86) => f.x86(),
            ResolvedFilter::Bcj(options::FILTER_POWERPC) => f.powerpc(),
            ResolvedFilter::Bcj(options::FILTER_IA64) => f.ia64(),
            ResolvedFilter::Bcj(options::FILTER_ARM) => f.arm(),
            ResolvedFilter::Bcj(options::FILTER_ARMTHUMB) => f.arm_thumb(),
            ResolvedFilter::Bcj(options::FILTER_SPARC) => f.sparc(),
            ResolvedFilter::Bcj(id) => unreachable!("resolve() only yields known BCJ ids, got {id:#x}"),
            ResolvedFilter::Lzma1(o) => f.lzma1(&xz2_lzma_options(o)?),
            ResolvedFilter::Lzma2(o) => f.lzma2(&xz2_lzma_options(o)?),
        };
    }
    Ok(f)
}

/// Raw (container-less) encode or decode of `input` with liblzma's raw coder.
fn raw_code(input: &[u8], chain: &[ResolvedFilter], encode: bool, out: &mut Vec<u8>) -> io::Result<()> {
    use lzma_sys::*;
    use std::ptr::null_mut;

    let (bcj, last) = options::split_chain(chain)?;
    let (id, o) = match last {
        ResolvedFilter::Lzma1(o) => (LZMA_FILTER_LZMA1, o),
        ResolvedFilter::Lzma2(o) => (LZMA_FILTER_LZMA2, o),
        ResolvedFilter::Bcj(_) => unreachable!("split_chain guarantees an LZMA tail"),
    };
    // SAFETY: lzma_options_lzma is a plain C struct; zeroed then filled by
    // lzma_lzma_preset, which validates the preset.
    let mut lzma: lzma_options_lzma = unsafe { std::mem::zeroed() };
    if unsafe { lzma_lzma_preset(&mut lzma, o.preset) } != 0 {
        return Err(io::Error::new(io::ErrorKind::InvalidInput, format!("xz: invalid preset {:#x}", o.preset)));
    }
    lzma.dict_size = o.dict_size;
    lzma.lc = o.lc;
    lzma.lp = o.lp;
    lzma.pb = o.pb;
    lzma.nice_len = o.nice_len;
    lzma.depth = o.depth;
    lzma.mode = match o.mode {
        Mode::Fast => LZMA_MODE_FAST,
        Mode::Normal => LZMA_MODE_NORMAL,
    };
    lzma.mf = match o.mf {
        MatchFinder::HashChain3 => LZMA_MF_HC3,
        MatchFinder::HashChain4 => LZMA_MF_HC4,
        MatchFinder::BinaryTree2 => LZMA_MF_BT2,
        MatchFinder::BinaryTree3 => LZMA_MF_BT3,
        MatchFinder::BinaryTree4 => LZMA_MF_BT4,
    };
    let mut filters: Vec<lzma_filter> = bcj.iter().map(|&id| lzma_filter { id, options: null_mut() }).collect();
    filters.push(lzma_filter { id, options: &mut lzma as *mut lzma_options_lzma as *mut _ });
    filters.push(lzma_filter { id: LZMA_VLI_UNKNOWN, options: null_mut() });

    /// Frees the coder on every exit path.
    struct Coder(lzma_stream);
    impl Drop for Coder {
        fn drop(&mut self) {
            unsafe { lzma_end(&mut self.0) }
        }
    }
    // SAFETY: an all-zero lzma_stream is LZMA_STREAM_INIT. `filters` and
    // `lzma` outlive the init call, which copies what it needs.
    let mut coder = Coder(unsafe { std::mem::zeroed() });
    let ret = unsafe {
        if encode {
            lzma_raw_encoder(&mut coder.0, filters.as_ptr())
        } else {
            lzma_raw_decoder(&mut coder.0, filters.as_ptr())
        }
    };
    if ret != LZMA_OK {
        return Err(lzma_error(ret));
    }
    coder.0.next_in = input.as_ptr();
    coder.0.avail_in = input.len();
    loop {
        out.reserve(input.len().max(64 << 10));
        let spare = out.spare_capacity_mut();
        let spare_len = spare.len();
        coder.0.next_out = spare.as_mut_ptr() as *mut u8;
        coder.0.avail_out = spare_len;
        // SAFETY: next_in/avail_in describe `input`; next_out/avail_out
        // describe `out`'s spare capacity, of which liblzma initializes the
        // first `spare_len - avail_out` bytes.
        let ret = unsafe { lzma_code(&mut coder.0, LZMA_FINISH) };
        let produced = spare_len - coder.0.avail_out;
        unsafe { out.set_len(out.len() + produced) };
        match ret {
            LZMA_STREAM_END => return Ok(()),
            LZMA_OK => continue,
            e => return Err(lzma_error(e)),
        }
    }
}

fn lzma_error(ret: lzma_sys::lzma_ret) -> io::Error {
    use lzma_sys::*;
    let (kind, msg) = match ret {
        LZMA_BUF_ERROR => (io::ErrorKind::UnexpectedEof, "xz: truncated raw stream"),
        LZMA_DATA_ERROR => (io::ErrorKind::InvalidData, "xz: corrupt raw stream"),
        LZMA_MEM_ERROR => (io::ErrorKind::OutOfMemory, "xz: out of memory"),
        LZMA_OPTIONS_ERROR => (io::ErrorKind::InvalidInput, "xz: unsupported filter options"),
        _ => (io::ErrorKind::Other, "xz: liblzma error"),
    };
    io::Error::new(kind, format!("{msg} (lzma_ret {ret})"))
}
