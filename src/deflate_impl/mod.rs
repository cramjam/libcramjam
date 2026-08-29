//! Pure-Rust DEFLATE / gzip / zlib implementation.
//!
//! Implements RFC 1951 (DEFLATE), RFC 1952 (gzip), and RFC 1950 (zlib).

pub mod adler32;
mod bitreader;
mod bitwriter;
pub mod compress;
pub mod crc32;
mod huffman;
pub mod inflate;
mod tables;

use std::io::{self, Read, Write};

pub const DEFAULT_COMPRESSION_LEVEL: u32 = 6;

// ---------------------------------------------------------------------------
// Raw DEFLATE
// ---------------------------------------------------------------------------

/// Compress raw DEFLATE.
pub fn deflate_compress<W: Write + ?Sized, R: Read>(
    mut input: R,
    output: &mut W,
    level: Option<u32>,
) -> io::Result<usize> {
    let level = level.unwrap_or(DEFAULT_COMPRESSION_LEVEL);
    let mut data = Vec::new();
    input.read_to_end(&mut data)?;
    let compressed = compress::deflate(&data, level);
    output.write_all(&compressed)?;
    Ok(compressed.len())
}

/// Decompress raw DEFLATE.
pub fn deflate_decompress<W: Write + ?Sized, R: Read>(
    mut input: R,
    output: &mut W,
) -> io::Result<usize> {
    let mut data = Vec::new();
    input.read_to_end(&mut data)?;
    crate::with_scratch(output, |decompressed| {
        decompressed.reserve(std::cmp::max(data.len().saturating_mul(4), 32768));
        inflate::inflate_into(&data, decompressed).map(|_| ())
    })
}

/// Worst-case compressed size for raw DEFLATE.
pub fn deflate_compress_bound(input_len: usize) -> usize {
    // Stored blocks: 5-byte overhead per block, MIN_BLOCK_LENGTH = 5000.
    const MIN_BLOCK: usize = 5000;
    let max_blocks = std::cmp::max((input_len + MIN_BLOCK - 1) / MIN_BLOCK, 1);
    5 * max_blocks + input_len
}

// ---------------------------------------------------------------------------
// Gzip (RFC 1952)
// ---------------------------------------------------------------------------

const GZIP_HEADER: [u8; 10] = [
    0x1F, 0x8B, // ID1, ID2
    0x08, // CM = deflate
    0x00, // FLG = none
    0x00, 0x00, 0x00, 0x00, // MTIME = 0
    0x00, // XFL
    0xFF, // OS = unknown
];
/// Minimum gzip framing overhead: 10-byte header + 8-byte footer.
pub const GZIP_MIN_OVERHEAD: usize = 10 + 8;

/// Compress gzip.
pub fn gzip_compress<W: Write + ?Sized, R: Read>(
    mut input: R,
    output: &mut W,
    level: Option<u32>,
) -> io::Result<usize> {
    let level = level.unwrap_or(DEFAULT_COMPRESSION_LEVEL);
    let mut data = Vec::new();
    input.read_to_end(&mut data)?;

    let crc = crc32::crc32(&data);
    let isize = (data.len() as u32).to_le_bytes();

    let deflated = compress::deflate(&data, level);

    let total = GZIP_HEADER.len() + deflated.len() + 8;
    let mut out = Vec::with_capacity(total);
    out.extend_from_slice(&GZIP_HEADER);
    out.extend_from_slice(&deflated);
    out.extend_from_slice(&crc.to_le_bytes());
    out.extend_from_slice(&isize);

    output.write_all(&out)?;
    Ok(out.len())
}

/// Decompress gzip (handles concatenated members).
pub fn gzip_decompress<W: Write + ?Sized, R: Read>(
    mut input: R,
    output: &mut W,
) -> io::Result<usize> {
    let mut data = Vec::new();
    input.read_to_end(&mut data)?;

    // Use the ISIZE hint from the last 4 bytes of the gzip footer to pre-allocate.
    // For single-member streams this gives the exact size; for multi-member
    // it's a lower bound (and we'll grow as needed).
    let size_hint = if data.len() >= 8 {
        u32::from_le_bytes(data[data.len() - 4..].try_into().unwrap()) as usize
    } else {
        data.len().saturating_mul(3)
    };
    crate::with_scratch(output, |decompressed| {
    decompressed.reserve(size_hint);
    let mut pos = 0;

    while pos < data.len() {
        // Header.
        if data.len() - pos < 10 {
            if pos == 0 {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidData,
                    "gzip: truncated header",
                ));
            }
            break;
        }
        if data[pos] != 0x1F || data[pos + 1] != 0x8B {
            if pos == 0 {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidData,
                    "gzip: invalid magic number",
                ));
            }
            break;
        }
        if data[pos + 2] != 8 {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "gzip: unsupported compression method",
            ));
        }
        let flg = data[pos + 3];
        if flg & 0xE0 != 0 {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "gzip: reserved flag bits set",
            ));
        }

        let mut hdr_end = pos + 10;
        if flg & 0x04 != 0 {
            if hdr_end + 2 > data.len() {
                return Err(io::Error::new(io::ErrorKind::InvalidData, "gzip: truncated FEXTRA"));
            }
            let xlen = u16::from_le_bytes([data[hdr_end], data[hdr_end + 1]]) as usize;
            hdr_end += 2 + xlen;
        }
        if flg & 0x08 != 0 {
            while hdr_end < data.len() && data[hdr_end] != 0 { hdr_end += 1; }
            hdr_end += 1;
        }
        if flg & 0x10 != 0 {
            while hdr_end < data.len() && data[hdr_end] != 0 { hdr_end += 1; }
            hdr_end += 1;
        }
        if flg & 0x02 != 0 { hdr_end += 2; }

        if hdr_end > data.len() {
            return Err(io::Error::new(io::ErrorKind::InvalidData, "gzip: truncated header fields"));
        }

        // Inflate, then CRC32 the result (separate pass — keeps inflate loop
        // unperturbed for maximum throughput; data is still in L2 cache).
        let out_start = decompressed.len();
        let consumed = inflate::inflate_into(&data[hdr_end..], decompressed)?;
        let data_end = hdr_end + consumed;
        let member_len = decompressed.len() - out_start;
        let actual_crc = crc32::crc32(&decompressed[out_start..]);

        // Footer: CRC32 + ISIZE.
        if data_end + 8 > data.len() {
            return Err(io::Error::new(io::ErrorKind::InvalidData, "gzip: truncated footer"));
        }
        let expected_crc =
            u32::from_le_bytes(data[data_end..data_end + 4].try_into().unwrap());
        let expected_isize =
            u32::from_le_bytes(data[data_end + 4..data_end + 8].try_into().unwrap());

        if actual_crc != expected_crc {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!("gzip: CRC-32 mismatch (expected {:08x}, got {:08x})", expected_crc, actual_crc),
            ));
        }
        if member_len as u32 != expected_isize {
            return Err(io::Error::new(io::ErrorKind::InvalidData, "gzip: ISIZE mismatch"));
        }

        pos = data_end + 8;
    }
    Ok(())
    })
}

/// Worst-case compressed size for gzip.
pub fn gzip_compress_bound(input_len: usize) -> usize {
    GZIP_MIN_OVERHEAD + deflate_compress_bound(input_len)
}

// ---------------------------------------------------------------------------
// Zlib (RFC 1950)
// ---------------------------------------------------------------------------

const ZLIB_HEADER_SIZE: usize = 2;
/// Minimum zlib framing overhead: 2-byte header + 4-byte Adler-32 footer.
pub const ZLIB_MIN_OVERHEAD: usize = ZLIB_HEADER_SIZE + 4;

/// Compute a valid 2-byte zlib header.
fn zlib_header(level: u32) -> [u8; 2] {
    let cmf: u8 = 0x78; // CM=8, CINFO=7 (32K window)
    let flevel: u8 = match level {
        0 | 1 => 0,
        2..=5 => 1,
        6 => 2,
        _ => 3,
    };
    let mut flg = flevel << 6;
    let check = (cmf as u16 * 256 + flg as u16) % 31;
    if check > 0 {
        flg += (31 - check) as u8;
    }
    [cmf, flg]
}

/// Compress zlib.
pub fn zlib_compress<W: Write + ?Sized, R: Read>(
    mut input: R,
    output: &mut W,
    level: Option<u32>,
) -> io::Result<usize> {
    let level = level.unwrap_or(DEFAULT_COMPRESSION_LEVEL);
    let mut data = Vec::new();
    input.read_to_end(&mut data)?;

    let checksum = adler32::adler32(&data);
    let deflated = compress::deflate(&data, level);
    let header = zlib_header(level);

    let total = 2 + deflated.len() + 4;
    let mut out = Vec::with_capacity(total);
    out.extend_from_slice(&header);
    out.extend_from_slice(&deflated);
    out.extend_from_slice(&checksum.to_be_bytes()); // big-endian!

    output.write_all(&out)?;
    Ok(out.len())
}

/// Decompress zlib.
pub fn zlib_decompress<W: Write + ?Sized, R: Read>(
    mut input: R,
    output: &mut W,
) -> io::Result<usize> {
    let mut data = Vec::new();
    input.read_to_end(&mut data)?;

    if data.len() < 6 {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "zlib: input too short",
        ));
    }

    let cmf = data[0];
    let flg = data[1];

    let cm = cmf & 0x0F;
    if cm != 8 {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "zlib: unsupported compression method",
        ));
    }
    let cinfo = cmf >> 4;
    if cinfo > 7 {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "zlib: invalid window size",
        ));
    }
    if (cmf as u16 * 256 + flg as u16) % 31 != 0 {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "zlib: header check failed",
        ));
    }
    let fdict = (flg >> 5) & 1;
    if fdict != 0 {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "zlib: preset dictionary not supported",
        ));
    }

    let hdr_size = ZLIB_HEADER_SIZE;
    crate::with_scratch(output, |decompressed| {
    decompressed.reserve(std::cmp::max(data.len().saturating_mul(4), 32768));
    let consumed = inflate::inflate_into(&data[hdr_size..], decompressed)?;
    let data_end = hdr_size + consumed;
    let actual = adler32::adler32(&decompressed);

    if data_end + 4 > data.len() {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "zlib: truncated adler-32",
        ));
    }
    let expected = u32::from_be_bytes(data[data_end..data_end + 4].try_into().unwrap());
    if actual != expected {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!("zlib: adler-32 mismatch (expected {:08x}, got {:08x})", expected, actual),
        ));
    }
    Ok(())
    })
}

/// Worst-case compressed size for zlib.
pub fn zlib_compress_bound(input_len: usize) -> usize {
    ZLIB_MIN_OVERHEAD + deflate_compress_bound(input_len)
}

// ---------------------------------------------------------------------------
// Streaming Write-adapters for the C API
// ---------------------------------------------------------------------------
//
// Each of these is a `Write`-shaped wrapper that buffers all input then
// compresses on `finish`.  They're generic over `W: Write` so callers can
// pass either a `Vec<u8>` directly or a `Cursor<Vec<u8>>` (the cramjam
// Python wrapper does the latter so it can call `into_inner()` to recover
// the raw vec from the returned cursor).

/// Write-adapter that accumulates input and produces gzip output on
/// [`finish`](GzipStreamCompressor::finish).
pub struct GzipStreamCompressor<W: Write = Vec<u8>> {
    input: Vec<u8>,
    output: W,
    level: u32,
}

impl<W: Write> GzipStreamCompressor<W> {
    pub fn new(output: W, level: u32) -> Self {
        Self {
            input: Vec::new(),
            output,
            level,
        }
    }

    pub fn get_ref(&self) -> &W {
        &self.output
    }

    pub fn finish(mut self) -> io::Result<W> {
        let mut buf = Vec::with_capacity(self.input.len() / 2);
        gzip_compress(&mut std::io::Cursor::new(self.input), &mut buf, Some(self.level))?;
        self.output.write_all(&buf)?;
        Ok(self.output)
    }
}

impl<W: Write> Write for GzipStreamCompressor<W> {
    fn write(&mut self, buf: &[u8]) -> io::Result<usize> {
        self.input.extend_from_slice(buf);
        Ok(buf.len())
    }
    fn flush(&mut self) -> io::Result<()> {
        Ok(())
    }
}

/// Write-adapter that accumulates input and produces raw deflate output
/// on [`finish`](DeflateStreamCompressor::finish).
pub struct DeflateStreamCompressor<W: Write = Vec<u8>> {
    input: Vec<u8>,
    output: W,
    level: u32,
}

impl<W: Write> DeflateStreamCompressor<W> {
    pub fn new(output: W, level: u32) -> Self {
        Self {
            input: Vec::new(),
            output,
            level,
        }
    }

    pub fn get_ref(&self) -> &W {
        &self.output
    }

    pub fn finish(mut self) -> io::Result<W> {
        let mut buf = Vec::with_capacity(self.input.len() / 2);
        deflate_compress(&mut std::io::Cursor::new(self.input), &mut buf, Some(self.level))?;
        self.output.write_all(&buf)?;
        Ok(self.output)
    }
}

impl<W: Write> Write for DeflateStreamCompressor<W> {
    fn write(&mut self, buf: &[u8]) -> io::Result<usize> {
        self.input.extend_from_slice(buf);
        Ok(buf.len())
    }
    fn flush(&mut self) -> io::Result<()> {
        Ok(())
    }
}

/// Write-adapter that accumulates input and produces zlib output on
/// [`finish`](ZlibStreamCompressor::finish).
pub struct ZlibStreamCompressor<W: Write = Vec<u8>> {
    input: Vec<u8>,
    output: W,
    level: u32,
}

impl<W: Write> ZlibStreamCompressor<W> {
    pub fn new(output: W, level: u32) -> Self {
        Self {
            input: Vec::new(),
            output,
            level,
        }
    }

    pub fn get_ref(&self) -> &W {
        &self.output
    }

    pub fn finish(mut self) -> io::Result<W> {
        let mut buf = Vec::with_capacity(self.input.len() / 2);
        zlib_compress(&mut std::io::Cursor::new(self.input), &mut buf, Some(self.level))?;
        self.output.write_all(&buf)?;
        Ok(self.output)
    }
}

impl<W: Write> Write for ZlibStreamCompressor<W> {
    fn write(&mut self, buf: &[u8]) -> io::Result<usize> {
        self.input.extend_from_slice(buf);
        Ok(buf.len())
    }
    fn flush(&mut self) -> io::Result<()> {
        Ok(())
    }
}
