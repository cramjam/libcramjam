//! Pure-Rust bzip2 implementation (RFC: bzip2 has no RFC, but the format is
//! documented at <https://en.wikipedia.org/wiki/Bzip2#File_format> and the
//! reference implementation is `libbzip2`).

pub mod bits;
pub mod crc;
pub mod decode;
pub mod encode;

use std::io::{self, Write};

/// Streaming Write-adapter for the C-API.  Buffers all input then encodes
/// once on `finish`.  Generic over `W: Write` so callers can pass either
/// a `Vec<u8>` directly or a `Cursor<Vec<u8>>` (the cramjam Python wrapper
/// uses the latter so it can call `into_inner()` on the returned cursor).
pub struct Bzip2StreamCompressor<W: Write = Vec<u8>> {
    input: Vec<u8>,
    output: W,
    level: u32,
}

impl<W: Write> Bzip2StreamCompressor<W> {
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

    /// Consume the compressor and return the underlying writer after the
    /// final compressed bytes have been written to it.
    pub fn finish(mut self) -> io::Result<W> {
        let compressed = encode::encode_stream(&self.input, self.level);
        self.output.write_all(&compressed)?;
        Ok(self.output)
    }
}

impl<W: Write> Write for Bzip2StreamCompressor<W> {
    fn write(&mut self, buf: &[u8]) -> io::Result<usize> {
        self.input.extend_from_slice(buf);
        Ok(buf.len())
    }

    fn flush(&mut self) -> io::Result<()> {
        Ok(())
    }
}
