//! lz4 de/compression interface — C backend (liblz4 via the `lz4` crate).
use std::cell::RefCell;
use std::io::{self, BufReader, Cursor, Error, Read, Write};

pub const BACKEND: crate::Backend = crate::Backend::C;

pub const DEFAULT_COMPRESSION_LEVEL: u32 = 4;
pub const LZ4_ACCELERATION_MAX: u32 = 65537;

/// Decompress lz4 frame data. Concatenated frames (and skippable frames) are
/// decoded in sequence; input that ends mid-frame is an error.
///
/// Drives `LZ4F_decompress` directly: `lz4::Decoder` stops after the first
/// frame and reads ahead, so the rest of a concatenated stream was silently
/// dropped (libcramjam <= 0.8).
pub fn decompress<W: Write + ?Sized, R: Read>(mut input: R, output: &mut W) -> Result<usize, Error> {
    use lz4::liblz4::*;

    struct Ctx(LZ4FDecompressionContext);
    impl Drop for Ctx {
        fn drop(&mut self) {
            unsafe { LZ4F_freeDecompressionContext(LZ4FDecompressionContext(self.0 .0)) };
        }
    }
    fn check(code: usize) -> io::Result<usize> {
        if unsafe { LZ4F_isError(code) } != 0 {
            let name = unsafe { std::ffi::CStr::from_ptr(LZ4F_getErrorName(code)) };
            return Err(io::Error::new(io::ErrorKind::InvalidData, format!("lz4: {}", name.to_string_lossy())));
        }
        Ok(code)
    }

    let mut data = Vec::new();
    input.read_to_end(&mut data)?;
    if data.is_empty() {
        return Err(io::Error::new(io::ErrorKind::UnexpectedEof, "lz4: empty input"));
    }
    let mut ctx = Ctx(LZ4FDecompressionContext(std::ptr::null_mut()));
    check(unsafe { LZ4F_createDecompressionContext(&mut ctx.0, LZ4F_VERSION) })?;
    let mut buf = vec![0u8; 256 << 10];
    let (mut pos, mut total) = (0usize, 0usize);
    loop {
        let mut src_size = data.len() - pos;
        let mut dst_size = buf.len();
        // SAFETY: src/dst pointers and sizes describe live buffers; liblz4
        // writes back how much of each it used.
        let hint = check(unsafe {
            LZ4F_decompress(
                LZ4FDecompressionContext(ctx.0 .0),
                buf.as_mut_ptr(),
                &mut dst_size,
                data[pos..].as_ptr(),
                &mut src_size,
                std::ptr::null(),
            )
        })?;
        pos += src_size;
        output.write_all(&buf[..dst_size])?;
        total += dst_size;
        if hint == 0 && pos == data.len() {
            return Ok(total); // last frame complete (the context resets itself between frames)
        }
        if pos == data.len() && dst_size == 0 {
            return Err(io::Error::new(io::ErrorKind::UnexpectedEof, "lz4: truncated frame"));
        }
    }
}

/// Worst-case compressed size of `input_len` bytes for the frame
/// [`compress`] writes: `LZ4F_compressBound` (blocks + end mark + content
/// checksum, assuming worst-case buffering) plus the maximum frame header,
/// which `LZ4F_compressBound` leaves out.
#[inline(always)]
pub fn compress_bound(input_len: usize, level: Option<u32>) -> usize {
    const LZ4F_HEADER_SIZE_MAX: usize = 19;
    // SAFETY: plain C struct; all-zero is valid (every enum field has a 0 variant).
    let mut prefs: lz4::liblz4::LZ4FPreferences = unsafe { std::mem::zeroed() };
    prefs.compression_level = level.unwrap_or(DEFAULT_COMPRESSION_LEVEL);
    // `compress` uses EncoderBuilder's default content checksum.
    prefs.frame_info.content_checksum_flag = lz4::liblz4::ContentChecksum::ChecksumEnabled;
    LZ4F_HEADER_SIZE_MAX + unsafe { lz4::liblz4::LZ4F_compressBound(input_len, &prefs) }
}

/// Compress as an lz4 frame.
#[inline(always)]
pub fn compress<W: Write + ?Sized, R: Read>(input: R, output: &mut W, level: Option<u32>) -> Result<usize, Error> {
    // lz4::Encoder is Write-only, so encode into a buffer and copy it out.
    // No auto_flush (0.8 had it): it cut a block per 8 KiB `io::copy` chunk,
    // which overran `compress_bound` on incompressible input.
    let mut encoder = lz4::EncoderBuilder::new()
        .level(level.unwrap_or(DEFAULT_COMPRESSION_LEVEL))
        .favor_dec_speed(true)
        .build(vec![])?;
    io::copy(&mut BufReader::new(input), &mut encoder)?;
    let (w, r) = encoder.finish();
    r?;
    let nbytes = io::copy(&mut Cursor::new(w), output)?;
    Ok(nbytes as _)
}

/// Encoder sink the wrapper can drain through `lz4::Encoder::writer(&self)`,
/// since the `lz4` crate offers no `&mut` access to its writer.
struct Sink(RefCell<Vec<u8>>);

impl Write for Sink {
    fn write(&mut self, buf: &[u8]) -> io::Result<usize> {
        self.0.get_mut().extend_from_slice(buf);
        Ok(buf.len())
    }
    fn flush(&mut self) -> io::Result<()> {
        Ok(())
    }
}

/// Streaming lz4 frame compressor: `flush` ends the current block so
/// everything written so far decodes, `finish` writes the end mark (and
/// the content checksum, when enabled).
pub struct Lz4StreamCompressor<W: Write> {
    enc: lz4::Encoder<Sink>,
    output: W,
}

impl<W: Write> Lz4StreamCompressor<W> {
    pub fn new(output: W, level: u32) -> Self {
        Self::with_options(output, level, true, false)
    }

    /// `block_linked = false` emits independent blocks; `content_checksum`
    /// appends the xxhash32 content checksum.
    pub fn with_options(output: W, level: u32, block_linked: bool, content_checksum: bool) -> Self {
        let enc = lz4::EncoderBuilder::new()
            .level(level)
            .block_mode(if block_linked { lz4::BlockMode::Linked } else { lz4::BlockMode::Independent })
            .checksum(if content_checksum {
                lz4::ContentChecksum::ChecksumEnabled
            } else {
                lz4::ContentChecksum::NoChecksum
            })
            .build(Sink(RefCell::new(Vec::new())))
            // ponytail: only fails if liblz4 can't allocate its context
            .expect("lz4: failed to create a compression context");
        Self { enc, output }
    }

    pub fn get_ref(&self) -> &W {
        &self.output
    }

    pub fn get_mut(&mut self) -> &mut W {
        &mut self.output
    }

    fn drain(&mut self) -> io::Result<()> {
        let mut pending = self.enc.writer().0.borrow_mut();
        self.output.write_all(&pending)?;
        pending.clear();
        Ok(())
    }

    pub fn finish(mut self) -> io::Result<W> {
        self.drain()?;
        let (sink, r) = self.enc.finish();
        r?;
        self.output.write_all(&sink.0.into_inner())?;
        Ok(self.output)
    }
}

impl<W: Write> Write for Lz4StreamCompressor<W> {
    fn write(&mut self, buf: &[u8]) -> io::Result<usize> {
        let n = self.enc.write(buf)?;
        self.drain()?;
        Ok(n)
    }
    fn flush(&mut self) -> io::Result<()> {
        self.enc.flush()?;
        self.drain()?;
        self.output.flush()
    }
}

/// Block-format helpers (no frame wrapper).
pub mod block {
    use lz4::block::CompressionMode;
    use std::io::Error;

    const PREPEND_SIZE: bool = true;

    /// Worst-case compressed size for `n` bytes, plus the 4-byte length
    /// prefix when `prepend_size` (the default).
    #[inline(always)]
    pub fn compress_bound(input_len: usize, prepend_size: Option<bool>) -> usize {
        match lz4::block::compress_bound(input_len) {
            Ok(len) => {
                if prepend_size.unwrap_or(true) {
                    len + 4
                } else {
                    len
                }
            }
            Err(_) => 0,
        }
    }

    /// Decompress into Vec. Must have been compressed with prepended uncompressed size.
    #[inline(always)]
    pub fn decompress_vec(input: &[u8]) -> Result<Vec<u8>, Error> {
        if input.len() < 4 {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidInput,
                "Input not long enough",
            ));
        }
        let bytes: [u8; 4] = input[..4].try_into().unwrap();
        let len = u32::from_le_bytes(bytes);
        let mut buf = vec![0u8; len as usize];
        let nbytes = decompress_into(&input[4..], &mut buf, Some(false))?;
        buf.truncate(nbytes);
        Ok(buf)
    }

    /// `size_prepended == Some(true)` means the input begins with the 4-byte
    /// uncompressed length. `output` can be larger than needed, but not smaller.
    #[inline(always)]
    pub fn decompress_into(input: &[u8], output: &mut [u8], size_prepended: Option<bool>) -> Result<usize, Error> {
        let uncompressed_size = if size_prepended.is_some_and(|v| v) {
            None // decompress_to_buffer will read from prepended size
        } else {
            Some(output.len() as _)
        };
        let nbytes = lz4::block::decompress_to_buffer(input, uncompressed_size, output)?;
        Ok(nbytes)
    }

    /// Compress into a fresh Vec, optionally prepending the 4-byte length.
    /// `level` and `acceleration` are accepted for API compatibility; blocks
    /// are always compressed in lz4's default mode.
    #[inline(always)]
    pub fn compress_vec(
        input: &[u8],
        level: Option<u32>,
        acceleration: Option<i32>,
        prepend_size: Option<bool>,
    ) -> Result<Vec<u8>, Error> {
        let len = compress_bound(input.len(), prepend_size);
        let mut buffer = vec![0u8; len];
        let nbytes = compress_into(input, &mut buffer, level, acceleration, prepend_size)?;
        buffer.truncate(nbytes);
        Ok(buffer)
    }

    /// Compress into a pre-allocated buffer.
    #[inline(always)]
    pub fn compress_into(
        input: &[u8],
        output: &mut [u8],
        _level: Option<u32>,
        _acceleration: Option<i32>,
        prepend_size: Option<bool>,
    ) -> Result<usize, Error> {
        let prepend_size = prepend_size.unwrap_or(PREPEND_SIZE);
        let nbytes = lz4::block::compress_to_buffer(input, Some(CompressionMode::DEFAULT), prepend_size, output)?;
        Ok(nbytes)
    }

    #[cfg(test)]
    mod tests {
        use super::{compress_vec, decompress_into, decompress_vec};

        const DATA: &[u8; 14] = b"howdy neighbor";

        #[test]
        fn round_trip_store_size() {
            let compressed = compress_vec(DATA, None, None, Some(true)).unwrap();
            let decompressed = decompress_vec(&compressed).unwrap();
            assert_eq!(&decompressed, DATA);
        }
        #[test]
        fn round_trip_no_store_size() {
            let compressed = compress_vec(DATA, None, None, Some(false)).unwrap();
            assert!(decompress_vec(&compressed).is_err());

            let mut decompressed = vec![0u8; DATA.len()];
            decompress_into(&compressed, &mut decompressed, Some(false)).unwrap();
            assert_eq!(&decompressed, DATA);

            let mut decompressed = vec![0u8; DATA.len() + 5_000];
            let n = decompress_into(&compressed, &mut decompressed, Some(false)).unwrap();
            assert_eq!(&decompressed[..n], DATA);
        }
    }
}
