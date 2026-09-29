//! deflate de/compression interface — C backend (`flate2`, as in libcramjam 0.8).
use flate2::Compression;
use std::io::{self, Error, Read, Write};

pub const BACKEND: crate::Backend = crate::Backend::C;

pub const DEFAULT_COMPRESSION_LEVEL: u32 = 6;
pub const MIN_BLOCK_LENGTH: usize = 5_000;

/// Compression upper bound
// xref: https://github.com/ebiggers/libdeflate/blob/6bb493615b0ef35c98fc4aa4ec04f448788db6a5/lib/deflate_compress.c#L4081
pub fn compress_bound(input_len: usize) -> usize {
    let max_blocks = std::cmp::max(input_len.div_ceil(MIN_BLOCK_LENGTH), 1);
    (5 * max_blocks) + input_len
}

/// Decompress deflate data
#[inline(always)]
pub fn decompress<W: Write + ?Sized, R: Read>(input: R, output: &mut W) -> Result<usize, Error> {
    inflate(input, output, false)
}

/// Inflate one raw deflate (or, with `zlib_header`, zlib) stream, erroring
/// if the input ends before the stream does. (`flate2`'s `read` decoders
/// return `Ok` with whatever they decoded so far on truncated input.)
pub(crate) fn inflate<W: Write + ?Sized, R: Read>(mut input: R, output: &mut W, zlib_header: bool) -> io::Result<usize> {
    use flate2::{Decompress, FlushDecompress, Status};
    let mut d = Decompress::new(zlib_header);
    let mut inbuf = vec![0u8; 64 << 10];
    let mut outbuf = Vec::with_capacity(128 << 10);
    let (mut start, mut end, mut eof, mut total) = (0, 0, false, 0);
    loop {
        if start == end && !eof {
            start = 0;
            end = input.read(&mut inbuf)?;
            eof = end == 0;
        }
        outbuf.clear();
        let before = d.total_in();
        let flush = if eof { FlushDecompress::Finish } else { FlushDecompress::None };
        let status = d
            .decompress_vec(&inbuf[start..end], &mut outbuf, flush)
            .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))?;
        let consumed = (d.total_in() - before) as usize;
        start += consumed;
        output.write_all(&outbuf)?;
        total += outbuf.len();
        match status {
            Status::StreamEnd => return Ok(total),
            _ if eof && consumed == 0 && outbuf.is_empty() => {
                return Err(io::Error::new(io::ErrorKind::UnexpectedEof, "deflate: truncated stream"))
            }
            _ => {}
        }
    }
}

/// Compress deflate data
#[inline(always)]
pub fn compress<W: Write + ?Sized, R: Read>(
    input: R,
    output: &mut W,
    level: Option<u32>,
) -> Result<usize, Error> {
    let level = level.unwrap_or(DEFAULT_COMPRESSION_LEVEL);
    let mut encoder = flate2::read::DeflateEncoder::new(input, Compression::new(level));
    let n_bytes = io::copy(&mut encoder, output)?;
    Ok(n_bytes as usize)
}

/// Streaming compressor: `flush` makes everything written so far decodable
/// (`Z_SYNC_FLUSH`), `finish` ends the stream.
pub struct DeflateStreamCompressor<W: Write>(flate2::write::DeflateEncoder<W>);

impl<W: Write> DeflateStreamCompressor<W> {
    pub fn new(output: W, level: u32) -> Self {
        Self(flate2::write::DeflateEncoder::new(output, Compression::new(level)))
    }
    pub fn get_ref(&self) -> &W {
        self.0.get_ref()
    }
    pub fn get_mut(&mut self) -> &mut W {
        self.0.get_mut()
    }
    pub fn finish(self) -> io::Result<W> {
        self.0.finish()
    }
}

impl<W: Write> Write for DeflateStreamCompressor<W> {
    fn write(&mut self, buf: &[u8]) -> io::Result<usize> {
        self.0.write(buf)
    }
    fn flush(&mut self) -> io::Result<()> {
        self.0.flush()
    }
}
