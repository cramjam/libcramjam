//! Pure-Rust bzip2 implementation (RFC: bzip2 has no RFC, but the format is
//! documented at <https://en.wikipedia.org/wiki/Bzip2#File_format> and the
//! reference implementation is `libbzip2`).

pub mod bits;
pub mod crc;
pub mod decode;
pub mod encode;

use std::io::{self, Write};

/// Streaming Write-adapter for the C-API.  Buffers all input then encodes
/// once on `finish`.  This is the same shape as `ZstdStreamCompressor` and
/// `GzipStreamCompressor` so the C-API plumbing in `capi.rs` can swap C
/// bzip2's `BzEncoder` for ours without restructuring.
pub struct Bzip2StreamCompressor {
    input: Vec<u8>,
    output: Vec<u8>,
    level: u32,
}

impl Bzip2StreamCompressor {
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

    /// Consume the compressor and return the final compressed bytes.
    pub fn finish(self) -> io::Result<Vec<u8>> {
        let mut output = self.output;
        let compressed = encode::encode_stream(&self.input, self.level);
        output.extend_from_slice(&compressed);
        Ok(output)
    }
}

impl Write for Bzip2StreamCompressor {
    fn write(&mut self, buf: &[u8]) -> io::Result<usize> {
        self.input.extend_from_slice(buf);
        Ok(buf.len())
    }

    fn flush(&mut self) -> io::Result<()> {
        Ok(())
    }
}
