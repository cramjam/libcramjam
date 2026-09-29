//! Pure-Rust bzip2 implementation (RFC: bzip2 has no RFC, but the format is
//! documented at <https://en.wikipedia.org/wiki/Bzip2#File_format> and the
//! reference implementation is `libbzip2`).

pub mod bits;
pub mod crc;
pub mod decode;
pub mod encode;

use std::io::{self, Write};

/// Streaming bzip2 compressor (`BZ2_bzCompress` with `BZ_RUN` / `BZ_FLUSH` /
/// `BZ_FINISH` semantics): input is compressed a block at a time as it
/// arrives; `flush` compresses whatever is pending as a block and writes
/// every whole byte out (a streaming decoder can then decode all data
/// written so far — the bit-level remainder stays queued, blocks are not
/// byte-aligned); `finish` writes the end-of-stream marker + combined CRC.
/// Memory: one pending block (≤ level × 100 kB) plus the bit writer.
pub struct Bzip2StreamCompressor<W: Write = Vec<u8>> {
    output: W,
    block_size: usize,
    nblock_max: usize,
    pending: Vec<u8>,
    bw: bits::BitWriter,
    combined_crc: u32,
}

impl<W: Write> Bzip2StreamCompressor<W> {
    pub fn new(output: W, level: u32) -> Self {
        let level = level.clamp(1, 9);
        let block_size = (level as usize) * 100_000;
        let mut bw = bits::BitWriter::new();
        encode::write_stream_header(&mut bw, level);
        Self {
            output,
            block_size,
            nblock_max: block_size - 19,
            pending: Vec::new(),
            bw,
            combined_crc: 0,
        }
    }

    pub fn get_ref(&self) -> &W {
        &self.output
    }

    pub fn get_mut(&mut self) -> &mut W {
        &mut self.output
    }

    /// Encode pending input into blocks: full blocks only, or everything
    /// when `all`.
    fn encode_pending(&mut self, all: bool) {
        while !self.pending.is_empty() && (all || self.pending.len() >= self.block_size) {
            let (rle1_out, consumed) = encode::forward_rle1_capped(&self.pending, self.nblock_max);
            debug_assert!(consumed > 0);
            let crc = encode::compute_block_crc(&self.pending[..consumed]);
            self.combined_crc = self.combined_crc.rotate_left(1) ^ crc;
            encode::encode_block_from_rle1(&mut self.bw, rle1_out, crc);
            self.pending.drain(..consumed);
        }
    }

    /// Encode pending input, write the end-of-stream marker and return the
    /// sink.
    pub fn finish(mut self) -> io::Result<W> {
        self.encode_pending(true);
        self.bw.write_bits(encode::EOS_MAGIC, 48);
        self.bw.write_bits(self.combined_crc as u64, 32);
        self.bw.align_to_byte();
        let bytes = self.bw.take_whole_bytes();
        self.output.write_all(&bytes)?;
        Ok(self.output)
    }
}

impl<W: Write> Write for Bzip2StreamCompressor<W> {
    fn write(&mut self, buf: &[u8]) -> io::Result<usize> {
        self.pending.extend_from_slice(buf);
        self.encode_pending(false);
        Ok(buf.len())
    }

    fn flush(&mut self) -> io::Result<()> {
        self.encode_pending(true);
        let bytes = self.bw.take_whole_bytes();
        self.output.write_all(&bytes)?;
        self.output.flush()
    }
}
