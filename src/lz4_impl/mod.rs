//! Pure-Rust LZ4 implementation (frame + block formats).

pub mod block;
pub mod frame;

use std::io::{self, Read, Write};

/// Streaming Write-adapter for the C-API.  Generic over `W: Write` so
/// callers can pass either a `Vec<u8>` or `Cursor<Vec<u8>>`.
pub struct Lz4StreamCompressor<W: Write = Vec<u8>> {
    input: Vec<u8>,
    output: W,
    level: u32,
}

impl<W: Write> Lz4StreamCompressor<W> {
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
        let frame = frame::encode_frame(&self.input);
        self.output.write_all(&frame)?;
        let _ = self.level; // level reserved for a future tunable encoder
        Ok(self.output)
    }
}

impl<W: Write> Write for Lz4StreamCompressor<W> {
    fn write(&mut self, buf: &[u8]) -> io::Result<usize> {
        self.input.extend_from_slice(buf);
        Ok(buf.len())
    }

    fn flush(&mut self) -> io::Result<()> {
        Ok(())
    }
}

/// Compress an input stream as an LZ4 frame and write it to `output`.
pub fn compress<W: Write + ?Sized, R: Read>(
    mut input: R,
    output: &mut W,
    _level: Option<u32>,
) -> io::Result<usize> {
    let mut data = Vec::new();
    input.read_to_end(&mut data)?;
    let frame = frame::encode_frame(&data);
    output.write_all(&frame)?;
    Ok(frame.len())
}

/// Decompress an LZ4 frame from `input` into `output`.
pub fn decompress<W: Write + ?Sized, R: Read>(
    mut input: R,
    output: &mut W,
) -> io::Result<usize> {
    let mut data = Vec::new();
    input.read_to_end(&mut data)?;
    let mut decoded = Vec::new();
    let mut consumed = 0usize;
    while consumed < data.len() {
        let n = frame::decode_frame(&data[consumed..], &mut decoded)?;
        if n == 0 {
            break;
        }
        consumed += n;
    }
    output.write_all(&decoded)?;
    Ok(decoded.len())
}

/// Worst-case compressed size.
pub fn compress_bound(input_len: usize, _level: Option<u32>) -> usize {
    // Frame overhead (~15 bytes header + per-block overhead) plus block bound.
    let blocks = (input_len + 65535) / 65536;
    15 + blocks * 8 + block::compress_bound(input_len)
}
