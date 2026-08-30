//! Pure-Rust LZ4 implementation (frame + block formats).

pub mod block;
pub mod frame;
pub mod hc;

use std::io::{self, Read, Write};

/// Streaming lz4 frame compressor (`LZ4F_compressUpdate` / `LZ4F_flush` /
/// `LZ4F_compressEnd` semantics): the frame header is emitted up front,
/// every full 64 KiB block is compressed as input arrives, `flush`
/// compresses whatever is pending as a (short) block and writes everything
/// produced so far to the sink, `finish` writes the end mark and the
/// optional xxhash32 content checksum. Linked blocks keep the match-finder
/// context across blocks. Memory is bounded: the input buffer is compacted
/// to the last 64 KiB of history once it grows past `COMPACT_AT`, and the
/// context is re-seeded from that history (`LZ4_loadDict`).
pub struct Lz4StreamCompressor<W: Write = Vec<u8>> {
    output: W,
    level: u32,
    block_linked: bool,
    content_checksum: bool,
    /// History + pending input; `buf[pending_start..]` is not yet compressed.
    buf: Vec<u8>,
    pending_start: usize,
    hc: Option<hc::HcCtx>,
    fast: Option<block::FastCtx>,
    /// Frame bytes produced but not yet written to `output`.
    out: Vec<u8>,
    compressed: Vec<u8>,
    xxh: frame::Xxh32,
}

const LZ4_BLOCK: usize = 64 << 10;
const LZ4_HISTORY: usize = 64 << 10;
const COMPACT_AT: usize = 1 << 20;

impl<W: Write> Lz4StreamCompressor<W> {
    pub fn new(output: W, level: u32) -> Self {
        Self::with_options(output, level, true, false)
    }

    /// `block_linked = false` emits independent blocks; `content_checksum`
    /// appends the xxhash32 content checksum (the C `LZ4F_preferences_t`
    /// knobs the Python wrapper exposes).
    pub fn with_options(output: W, level: u32, block_linked: bool, content_checksum: bool) -> Self {
        let mut out = Vec::with_capacity(LZ4_BLOCK + 64);
        frame::write_frame_header(&mut out, block_linked, content_checksum);
        let use_hc = frame::use_hc_for(Some(level));
        Self {
            output,
            level,
            block_linked,
            content_checksum,
            buf: Vec::new(),
            pending_start: 0,
            hc: if use_hc { Some(hc::HcCtx::new()) } else { None },
            fast: if use_hc { None } else { Some(block::FastCtx::new()) },
            out,
            compressed: Vec::with_capacity(block::compress_bound(LZ4_BLOCK)),
            xxh: frame::Xxh32::new(),
        }
    }

    pub fn get_ref(&self) -> &W {
        &self.output
    }

    pub fn get_mut(&mut self) -> &mut W {
        &mut self.output
    }

    fn compress_range(&mut self, start: usize, end: usize) {
        self.compressed.clear();
        if let Some(ctx) = self.hc.as_mut() {
            hc::compress_block_hc_continue(ctx, &self.buf, start, end, &mut self.compressed, self.level);
        } else {
            block::compress_block_fast_continue(self.fast.as_mut().unwrap(), &self.buf, start, end, &mut self.compressed);
        }
    }

    /// Compress `buf[pending_start..end]` as one block.
    fn emit_block(&mut self, end: usize) {
        let start = self.pending_start;
        if self.block_linked {
            self.compress_range(start, end);
        } else {
            // Independent block: no history at all (the parser may not even
            // extend a match backwards across the block start), so compress
            // the block as its own input, like `LZ4_compress_fast_extState`.
            self.compressed.clear();
            frame::compress_independent_block(&self.buf[start..end], self.level, self.hc.is_some(), &mut self.compressed);
        }
        frame::emit_block(&mut self.out, &self.compressed, &self.buf[start..end]);
        self.pending_start = end;
        self.compact();
    }

    /// Drop history beyond the last 64 KiB and re-seed the context from it.
    fn compact(&mut self) {
        if self.pending_start < COMPACT_AT {
            return;
        }
        let keep = if self.block_linked { LZ4_HISTORY.min(self.pending_start) } else { 0 };
        let drop = self.pending_start - keep;
        self.buf.drain(..drop);
        self.pending_start = keep;
        if self.hc.is_some() {
            self.hc = Some(hc::HcCtx::new());
        } else {
            self.fast = Some(block::FastCtx::new());
        }
        if keep > 0 {
            // Re-seed: run the parser over the retained history (output
            // discarded), which inserts it into the tables like
            // `LZ4_loadDict` / `LZ4_loadDictHC`.
            self.compress_range(0, keep);
        }
    }

    /// Emit every full block in the pending input.
    fn emit_full_blocks(&mut self) {
        while self.buf.len() - self.pending_start >= LZ4_BLOCK {
            let end = self.pending_start + LZ4_BLOCK;
            self.emit_block(end);
        }
    }

    fn write_out(&mut self) -> io::Result<()> {
        if !self.out.is_empty() {
            self.output.write_all(&self.out)?;
            self.out.clear();
        }
        Ok(())
    }

    /// Emit pending input, the end mark and the content checksum; return
    /// the sink.
    pub fn finish(mut self) -> io::Result<W> {
        self.emit_full_blocks();
        if self.buf.len() > self.pending_start {
            let end = self.buf.len();
            self.emit_block(end);
        }
        self.out.extend_from_slice(&0u32.to_le_bytes());
        if self.content_checksum {
            self.out.extend_from_slice(&self.xxh.finish().to_le_bytes());
        }
        self.write_out()?;
        Ok(self.output)
    }
}

impl<W: Write> Write for Lz4StreamCompressor<W> {
    fn write(&mut self, data: &[u8]) -> io::Result<usize> {
        self.buf.extend_from_slice(data);
        if self.content_checksum {
            self.xxh.update(data);
        }
        self.emit_full_blocks();
        Ok(data.len())
    }

    fn flush(&mut self) -> io::Result<()> {
        self.emit_full_blocks();
        if self.buf.len() > self.pending_start {
            let end = self.buf.len();
            self.emit_block(end);
        }
        self.write_out()?;
        self.output.flush()
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
