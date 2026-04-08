//! Pure-Rust LZ4 implementation (frame + block formats).

pub mod block;
pub mod frame;

use std::io::{self, Read, Write};

/// Streaming Write-adapter for the C-API.  Buffers all input then encodes
/// once on `finish`.  Mirrors `Bzip2StreamCompressor` and
/// `ZstdStreamCompressor` so the C-API plumbing in `capi.rs` can swap C
/// lz4's `Encoder` for ours without restructuring.
pub struct Lz4StreamCompressor {
    input: Vec<u8>,
    output: Vec<u8>,
    level: u32,
}

impl Lz4StreamCompressor {
    pub fn new(output: Vec<u8>, level: u32) -> Self {
        Self {
            input: Vec::new(),
            output,
            level,
        }
    }

    pub fn get_ref(&self) -> &Vec<u8> {
        &self.output
    }

    pub fn finish(self) -> io::Result<Vec<u8>> {
        let mut output = self.output;
        let frame = frame::encode_frame(&self.input);
        output.extend_from_slice(&frame);
        let _ = self.level; // level reserved for a future tunable encoder
        Ok(output)
    }
}

impl Write for Lz4StreamCompressor {
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
