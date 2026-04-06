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
    let mut decompressed = Vec::with_capacity(std::cmp::max(data.len().saturating_mul(4), 32768));
    inflate::inflate_into(&data, &mut decompressed)?;
    output.write_all(&decompressed)?;
    Ok(decompressed.len())
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
const GZIP_FOOTER_SIZE: usize = 8;
const GZIP_MIN_HEADER_SIZE: usize = 10;
pub const GZIP_MIN_OVERHEAD: usize = GZIP_MIN_HEADER_SIZE + GZIP_FOOTER_SIZE;

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
    let mut decompressed = Vec::with_capacity(size_hint);
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
        let consumed = inflate::inflate_into(&data[hdr_end..], &mut decompressed)?;
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

    let n = decompressed.len();
    output.write_all(&decompressed)?;
    Ok(n)
}

/// Worst-case compressed size for gzip.
pub fn gzip_compress_bound(input_len: usize) -> usize {
    GZIP_MIN_OVERHEAD + deflate_compress_bound(input_len)
}

// ---------------------------------------------------------------------------
// Zlib (RFC 1950)
// ---------------------------------------------------------------------------

const ZLIB_HEADER_SIZE: usize = 2;
const ZLIB_FOOTER_SIZE: usize = 4; // Adler-32
pub const ZLIB_MIN_OVERHEAD: usize = ZLIB_HEADER_SIZE + ZLIB_FOOTER_SIZE;

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
    let mut decompressed = Vec::with_capacity(std::cmp::max(data.len().saturating_mul(4), 32768));
    let consumed = inflate::inflate_into(&data[hdr_size..], &mut decompressed)?;
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

    let n = decompressed.len();
    output.write_all(&decompressed)?;
    Ok(n)
}

/// Worst-case compressed size for zlib.
pub fn zlib_compress_bound(input_len: usize) -> usize {
    ZLIB_MIN_OVERHEAD + deflate_compress_bound(input_len)
}

// ---------------------------------------------------------------------------
// Streaming gzip compressor (Write-adapter for the C API)
// ---------------------------------------------------------------------------

/// A Write-adapter that accumulates input and produces gzip output on
/// [`finish`](GzipStreamCompressor::finish).
///
/// This provides API compatibility with the `flate2::write::GzEncoder`
/// pattern used by the C API.  Intermediate calls to
/// [`get_ref`](GzipStreamCompressor::get_ref) return an empty buffer
/// until `finish` is called.
pub struct GzipStreamCompressor {
    input: Vec<u8>,
    output: Vec<u8>,
    level: u32,
}

impl GzipStreamCompressor {
    pub fn new(output: Vec<u8>, level: u32) -> Self {
        Self {
            input: Vec::new(),
            output,
            level,
        }
    }

    /// Peek at the output buffer.
    pub fn get_ref(&self) -> &Vec<u8> {
        &self.output
    }

    /// Finalize compression and return the gzip output.
    pub fn finish(self) -> io::Result<Vec<u8>> {
        let mut output = self.output;
        gzip_compress(
            &mut std::io::Cursor::new(self.input),
            &mut output,
            Some(self.level),
        )?;
        Ok(output)
    }
}

impl Write for GzipStreamCompressor {
    fn write(&mut self, buf: &[u8]) -> io::Result<usize> {
        self.input.extend_from_slice(buf);
        Ok(buf.len())
    }

    fn flush(&mut self) -> io::Result<()> {
        Ok(())
    }
}
