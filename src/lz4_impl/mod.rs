//! Pure-Rust LZ4 implementation (frame + block formats).

pub mod block;
pub mod frame;

use std::io::{self, Read, Write};

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
