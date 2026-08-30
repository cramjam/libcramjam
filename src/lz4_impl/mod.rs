//! Pure-Rust LZ4 implementation (frame + block formats).

pub mod block;
pub mod frame;
pub mod hc;

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
        let frame = frame::encode_frame_at(&self.input, Some(self.level));
        self.output.write_all(&frame)?;
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
///
/// `level` follows the lz4 frame-format convention: `0..=2` use the fast
/// hash-table parser, `3..=12` use the HC parser (chained hash table +
/// lazy match) — higher levels search deeper chains for better ratios.
pub fn compress<W: Write + ?Sized, R: Read>(
    mut input: R,
    output: &mut W,
    level: Option<u32>,
) -> io::Result<usize> {
    let mut data = Vec::new();
    input.read_to_end(&mut data)?;
    let frame = frame::encode_frame_at(&data, level);
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
    let mut sink = crate::SinkRef(output);
    crate::scratch_with(|buf| {
        let mut consumed = 0usize;
        let mut total = 0usize;
        while consumed < data.len() {
            let (n, produced) = frame::decode_frame_streaming(&data[consumed..], buf, Some(&mut sink))?;
            if n == 0 {
                break;
            }
            sink.0.write_all(buf)?;
            buf.clear();
            consumed += n;
            total += produced;
        }
        Ok(total)
    })
}

/// Worst-case compressed size.
pub fn compress_bound(input_len: usize, _level: Option<u32>) -> usize {
    // Frame overhead (~15 bytes header + per-block overhead) plus block bound.
    let blocks = (input_len + 65535) / 65536;
    15 + blocks * 8 + block::compress_bound(input_len)
}
