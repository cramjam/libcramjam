//! Zstandard frame encoder (RFC 8878), structured like C zstd's
//! `ZSTD_compress_frameChunk` → `ZSTD_compressBlock_internal`:
//!
//!   * per-level parameters from `cparams.rs` (C's level table);
//!   * the input is parsed in 128 KiB blocks by the strategy's match finder
//!     (`parse_fast.rs`: fast / doubleFast, `parse_lazy.rs`: row-hash
//!     greedy / lazy / lazy2), match-finder tables persisting across blocks;
//!   * each block's `SeqStore` is entropy-coded by `entropy.rs`
//!     (Huffman literals, FSE sequences) or emitted raw / RLE when that
//!     does not pay.

use std::cell::RefCell;

use super::cparams::{cparams, CParams, Strategy};
use super::entropy;
use super::parse_fast::{compress_block_dfast, compress_block_fast, BlockCtx, FastState};
use super::parse_lazy::{compress_block_lazy, RowState};
use super::seqstore::SeqStore;

const ZSTD_MAGIC: u32 = 0xFD2FB528;
const MAX_BLOCK_SIZE: usize = 128 * 1024;
/// Inputs above this are split into independent frames so positions fit
/// in the match finders' `u32` indices.
const MAX_FRAME_INPUT: usize = 1 << 30;

// =========================================================================
// Public entry points
// =========================================================================

/// Compress `input` into one zstd frame (or several concatenated frames
/// for inputs over 1 GiB).
///
/// `level`: 0 = raw blocks only. 1..=22 as in C zstd (levels above 12 use
/// the strongest lazy2 configuration).
pub fn encode_frame(input: &[u8], level: i32, _content_size: Option<u64>) -> Vec<u8> {
    let mut out = Vec::with_capacity(input.len() / 2 + 64);
    encode_frame_into(&mut out, input, level);
    out
}

/// [`encode_frame`] appending to a caller-provided buffer.
pub fn encode_frame_into(out: &mut Vec<u8>, input: &[u8], level: i32) {
    if input.len() > MAX_FRAME_INPUT {
        for chunk in input.chunks(MAX_FRAME_INPUT) {
            encode_one_frame(out, chunk, level);
        }
    } else {
        encode_one_frame(out, input, level);
    }
}

fn encode_one_frame(out: &mut Vec<u8>, input: &[u8], level: i32) {
    // -- Frame header --
    out.extend_from_slice(&ZSTD_MAGIC.to_le_bytes());
    // No content checksum, like C zstd's default (`ZSTD_c_checksumFlag`
    // = 0, which is what the previous C-backed cramjam emitted); the
    // xxhash64 pass was ~8% of L3 compress time.
    let content_checksum = false;
    let single_segment = true;
    let fcs_field_size = fcs_field_size_for(input.len() as u64);
    let descriptor: u8 = (fcs_field_size.flag << 6) | ((single_segment as u8) << 5) | ((content_checksum as u8) << 2);
    out.push(descriptor);
    write_fcs(out, input.len() as u64, fcs_field_size.bytes);

    // -- Blocks --
    if input.is_empty() {
        write_block_header(out, 0, BlockType::Raw, true);
    } else if level <= 0 {
        encode_raw_blocks(out, input);
    } else {
        SCRATCH.with(|s| encode_compressed_blocks(out, input, level, &mut s.borrow_mut()));
    }

    // -- Content checksum (xxhash64 lower 32 bits) --
    if content_checksum {
        let hash = super::decode::xxhash64_public(input, 0) as u32;
        out.extend_from_slice(&hash.to_le_bytes());
    }
}

/// Worst-case compressed size: raw blocks + frame overhead.
pub fn compress_bound(input_len: usize) -> usize {
    let num_blocks = (input_len + MAX_BLOCK_SIZE - 1) / MAX_BLOCK_SIZE.max(1);
    let num_frames = (input_len + MAX_FRAME_INPUT - 1) / MAX_FRAME_INPUT;
    (14 + 4) * num_frames.max(1) + num_blocks.max(1) * 3 + input_len
}

// =========================================================================
// Block framing
// =========================================================================

#[derive(Copy, Clone)]
#[repr(u8)]
enum BlockType {
    Raw = 0,
    Rle = 1,
    Compressed = 2,
}

fn write_block_header(out: &mut Vec<u8>, size: usize, ty: BlockType, last: bool) {
    let header = ((size as u32) << 3) | ((ty as u32) << 1) | (last as u32);
    out.push(header as u8);
    out.push((header >> 8) as u8);
    out.push((header >> 16) as u8);
}

fn encode_raw_blocks(out: &mut Vec<u8>, input: &[u8]) {
    let mut offset = 0;
    while offset < input.len() {
        let chunk = (input.len() - offset).min(MAX_BLOCK_SIZE);
        let last = offset + chunk >= input.len();
        write_block_header(out, chunk, BlockType::Raw, last);
        out.extend_from_slice(&input[offset..offset + chunk]);
        offset += chunk;
    }
}

// =========================================================================
// Compressed blocks
// =========================================================================

/// Per-thread reusable state: match-finder tables + the block seq store.
struct Scratch {
    fast: FastState,
    row: RowState,
    seq: SeqStore,
    payload: Vec<u8>,
}

thread_local! {
    static SCRATCH: RefCell<Scratch> = RefCell::new(Scratch {
        fast: FastState::new(),
        row: RowState::new(),
        seq: SeqStore::new(),
        payload: Vec::new(),
    });
}

fn is_rle(block: &[u8]) -> bool {
    let b = block[0];
    block.iter().all(|&x| x == b)
}

/// Everything a block encoder needs across blocks: the level's parameters,
/// the match-finder tables, the repeat offsets and the "first block" flag.
/// Shared by the one-shot frame loop and the streaming compressor so both
/// produce identical blocks for identical input.
struct BlockEncoder {
    params: CParams,
    block_size: usize,
    is_row: bool,
    rep: [u32; 3],
    first_block: bool,
}

impl BlockEncoder {
    fn new(params: CParams, scratch: &mut Scratch) -> Self {
        let block_size = MAX_BLOCK_SIZE.min(1usize << params.window_log);
        let is_row = params.strategy >= Strategy::Greedy;
        let row_log = params.search_log.clamp(4, 6);
        if is_row {
            scratch.row.reset(params.hash_log, row_log);
        } else {
            scratch.fast.reset(params.hash_log, params.chain_log, params.strategy == Strategy::Dfast);
        }
        Self { params, block_size, is_row, rep: [1, 4, 8], first_block: true }
    }

    /// Encode `buf[pos..pos + len]` as one block (raw / RLE / compressed)
    /// appended to `out`. `buf` is the whole addressable history (positions
    /// in the match-finder tables index into it); `avail_end` is the end of
    /// the data that may be read past the block (literal-copy / count limit).
    fn encode_block(&mut self, out: &mut Vec<u8>, buf: &[u8], pos: usize, len: usize, avail_end: usize, last: bool, scratch: &mut Scratch) {
        let params = &self.params;
        let block = &buf[pos..pos + len];
        let base = buf.as_ptr();
        let input_end = unsafe { base.add(avail_end) };

        // ZSTD_buildSeqStore: tiny blocks are not worth compressing.
        let compressed = if len < 2 + 3 + 1 + 1 {
            None
        } else {
            let saved_rep = self.rep;
            // limited update after a very long match (row finder)
            if self.is_row {
                let curr = pos as u32;
                let ntu = scratch.row.next_to_update;
                if curr > ntu + 384 {
                    scratch.row.next_to_update = curr - 192.min(curr - ntu - 384);
                }
            }
            scratch.seq.reset(len);
            let mut ctx = BlockCtx {
                base,
                istart: pos as u32,
                block_len: len,
                input_end,
                window_log: params.window_log,
                target_length: params.target_length,
                rep: &mut self.rep,
            };
            let last_lits = unsafe {
                match params.strategy {
                    Strategy::Fast => compress_block_fast(&mut scratch.fast, &mut scratch.seq, &mut ctx, params.hash_log, params.min_match),
                    Strategy::Dfast => {
                        compress_block_dfast(&mut scratch.fast, &mut scratch.seq, &mut ctx, params.hash_log, params.chain_log, params.min_match)
                    }
                    Strategy::Greedy => compress_block_lazy(&mut scratch.row, &mut scratch.seq, &mut ctx, params.search_log, params.min_match, 0),
                    Strategy::Lazy => compress_block_lazy(&mut scratch.row, &mut scratch.seq, &mut ctx, params.search_log, params.min_match, 1),
                    Strategy::Lazy2 => compress_block_lazy(&mut scratch.row, &mut scratch.seq, &mut ctx, params.search_log, params.min_match, 2),
                }
            };
            scratch.seq.store_last_literals(&block[len - last_lits..]);

            let payload = &mut scratch.payload;
            payload.clear();
            let nb_seq = scratch.seq.seqs.len();
            let lit_size = scratch.seq.lit.len();
            let suspect_uncompressible = nb_seq == 0 || lit_size / nb_seq.max(1) >= 20;
            entropy::compress_literals(payload, &scratch.seq.lit, params.strategy, suspect_uncompressible);
            entropy::compress_sequences(payload, &scratch.seq.seqs, params.strategy);
            // ZSTD_entropyCompressSeqStore: not compressible enough → raw.
            let max_c_size = len - entropy::min_gain(len);
            if payload.len() >= max_c_size {
                self.rep = saved_rep;
                None
            } else {
                Some(payload.len())
            }
        };

        match compressed {
            Some(c_size) => {
                if !self.first_block && c_size < 25 && is_rle(block) {
                    write_block_header(out, len, BlockType::Rle, last);
                    out.push(block[0]);
                } else {
                    write_block_header(out, c_size, BlockType::Compressed, last);
                    out.extend_from_slice(&scratch.payload);
                }
            }
            None => {
                write_block_header(out, len, BlockType::Raw, last);
                out.extend_from_slice(block);
            }
        }
        self.first_block = false;
    }

    /// Drop `shift` bytes of history from the front of the addressable
    /// buffer (the caller memmoves the buffer): rebase every table index
    /// like C's `ZSTD_reduceTable`. Entries that pointed below the new
    /// start saturate to 0 — a candidate that is only ever verified by byte
    /// comparison and the window-distance check, so a stale 0 is harmless.
    /// `shift` must be a multiple of the row hash-cache size so the cache's
    /// position→slot mapping stays valid.
    fn rebase(&mut self, shift: usize, scratch: &mut Scratch) {
        let shift = shift as u32;
        if self.is_row {
            for e in scratch.row.hash_table.iter_mut() {
                *e = e.saturating_sub(shift);
            }
            scratch.row.next_to_update = scratch.row.next_to_update.saturating_sub(shift);
        } else {
            for e in scratch.fast.hash_table.iter_mut() {
                *e = e.saturating_sub(shift);
            }
            for e in scratch.fast.hash_small.iter_mut() {
                *e = e.saturating_sub(shift);
            }
        }
    }
}

fn encode_compressed_blocks(out: &mut Vec<u8>, input: &[u8], level: i32, scratch: &mut Scratch) {
    let params: CParams = cparams(level, input.len());
    let mut enc = BlockEncoder::new(params, scratch);
    let mut pos = 0usize;
    while pos < input.len() {
        let len = (input.len() - pos).min(enc.block_size);
        let last = pos + len >= input.len();
        enc.encode_block(out, input, pos, len, input.len(), last, scratch);
        pos += len;
    }
}

// =========================================================================
// Streaming (ZSTD_compressStream2 semantics)
// =========================================================================

/// Streaming frame encoder: `push` buffers input and emits full blocks,
/// `flush` compresses whatever is pending into block(s) so a decoder can
/// consume everything written so far (`ZSTD_e_flush`), `end` writes the
/// last block (`ZSTD_e_end`). The frame header carries no content size
/// (Single_Segment = 0, window descriptor from the level's windowLog).
///
/// Memory is bounded: the addressable buffer holds at most about
/// `2 * window + block` bytes — once more than a window of history sits
/// behind the pending data the front is dropped and the tables rebased.
pub struct StreamEncoder {
    enc: Option<BlockEncoder>,
    scratch: Box<Scratch>,
    /// history + pending input; positions in the tables index into it.
    buf: Vec<u8>,
    /// `buf[pending..]` has not been encoded yet.
    pending: usize,
    window: usize,
    block_size: usize,
    header_written: bool,
    finished: bool,
}

impl StreamEncoder {
    pub fn new(level: i32) -> Self {
        let params = cparams(level, 0);
        let mut scratch = Box::new(Scratch {
            fast: FastState::new(),
            row: RowState::new(),
            seq: SeqStore::new(),
            payload: Vec::new(),
        });
        let (enc, window, block_size) = if level <= 0 {
            (None, MAX_BLOCK_SIZE, MAX_BLOCK_SIZE)
        } else {
            let e = BlockEncoder::new(params, &mut scratch);
            let bs = e.block_size;
            (Some(e), 1usize << params.window_log, bs)
        };
        Self {
            enc,
            scratch,
            buf: Vec::new(),
            pending: 0,
            window,
            block_size,
            header_written: false,
            finished: false,
        }
    }

    fn write_header(&mut self, out: &mut Vec<u8>) {
        if self.header_written {
            return;
        }
        self.header_written = true;
        out.extend_from_slice(&ZSTD_MAGIC.to_le_bytes());
        // FCS flag 0 + Single_Segment 0 ⇒ no content size; no checksum.
        out.push(0);
        // Window descriptor: exponent = windowLog - 10, mantissa 0.
        let window_log = self.window.trailing_zeros();
        out.push(((window_log - 10) as u8) << 3);
    }

    /// Emit one block of `len` pending bytes (`len > 0`).
    fn emit_block(&mut self, out: &mut Vec<u8>, len: usize, last: bool) {
        self.write_header(out);
        let pos = self.pending;
        match self.enc.as_mut() {
            Some(enc) => enc.encode_block(out, &self.buf, pos, len, self.buf.len(), last, &mut self.scratch),
            None => {
                write_block_header(out, len, BlockType::Raw, last);
                out.extend_from_slice(&self.buf[pos..pos + len]);
            }
        }
        self.pending += len;
    }

    /// Drop history beyond the window once it costs more than a window's
    /// worth of memory (keeps exactly `window` bytes behind the pending
    /// data afterwards).
    fn maybe_slide(&mut self) {
        if self.pending < 2 * self.window {
            return;
        }
        // Shift rounded down to a multiple of 64 (row hash-cache alignment).
        let shift = (self.pending - self.window) & !63;
        if shift == 0 {
            return;
        }
        self.buf.drain(..shift);
        self.pending -= shift;
        if let Some(enc) = self.enc.as_mut() {
            enc.rebase(shift, &mut self.scratch);
        }
    }

    /// Feed input; full blocks are emitted as they become available.
    pub fn push(&mut self, out: &mut Vec<u8>, data: &[u8]) {
        debug_assert!(!self.finished);
        self.buf.extend_from_slice(data);
        while self.buf.len() - self.pending >= self.block_size {
            self.emit_block(out, self.block_size, false);
        }
        self.maybe_slide();
    }

    /// `ZSTD_e_flush`: compress everything pending (in >= 1 blocks) so the
    /// output so far decodes to all input pushed so far.
    pub fn flush(&mut self, out: &mut Vec<u8>) {
        while self.buf.len() > self.pending {
            let len = (self.buf.len() - self.pending).min(self.block_size);
            self.emit_block(out, len, false);
        }
        self.maybe_slide();
    }

    /// `ZSTD_e_end`: everything pending, the last block flag set (an empty
    /// raw last block when nothing is pending), then the encoder is done.
    pub fn end(&mut self, out: &mut Vec<u8>) {
        if self.finished {
            return;
        }
        while self.buf.len() - self.pending > self.block_size {
            self.emit_block(out, self.block_size, false);
        }
        let rest = self.buf.len() - self.pending;
        if rest > 0 {
            self.emit_block(out, rest, true);
        } else {
            self.write_header(out);
            write_block_header(out, 0, BlockType::Raw, true);
        }
        self.finished = true;
        self.buf = Vec::new();
        self.pending = 0;
    }
}

// =========================================================================
// Frame Content Size encoding
// =========================================================================

struct FcsInfo {
    flag: u8,
    bytes: u8,
}

fn fcs_field_size_for(size: u64) -> FcsInfo {
    if size < 256 {
        FcsInfo { flag: 0, bytes: 1 }
    } else if size < 65536 + 256 {
        FcsInfo { flag: 1, bytes: 2 }
    } else if size <= u32::MAX as u64 {
        FcsInfo { flag: 2, bytes: 4 }
    } else {
        FcsInfo { flag: 3, bytes: 8 }
    }
}

fn write_fcs(out: &mut Vec<u8>, size: u64, bytes: u8) {
    match bytes {
        1 => out.push(size as u8),
        2 => out.extend_from_slice(&((size - 256) as u16).to_le_bytes()),
        4 => out.extend_from_slice(&(size as u32).to_le_bytes()),
        _ => out.extend_from_slice(&size.to_le_bytes()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn roundtrip(input: &[u8], level: i32) {
        let frame = encode_frame(input, level, Some(input.len() as u64));
        let mut decoded = Vec::new();
        let n = super::super::decode::decode_frame(&frame, &mut decoded).unwrap();
        assert_eq!(n, frame.len());
        assert_eq!(decoded, input, "level {level} len {}", input.len());
    }

    #[test]
    fn frame_roundtrip_text() {
        let input: Vec<u8> = b"The quick brown fox jumps over the lazy dog. ".repeat(40);
        for level in 1..=9 {
            roundtrip(&input, level);
        }
    }

    #[test]
    fn frame_roundtrip_random_falls_back_to_raw() {
        let mut s: u32 = 0xDEAD_BEEF;
        let input: Vec<u8> = (0..2048)
            .map(|_| {
                s ^= s << 13;
                s ^= s >> 17;
                s ^= s << 5;
                (s >> 16) as u8
            })
            .collect();
        for level in 1..=9 {
            roundtrip(&input, level);
        }
    }

    #[test]
    fn frame_roundtrip_sizes() {
        let mut s: u32 = 12345;
        for n in [0usize, 1, 2, 7, 8, 9, 15, 16, 17, 31, 63, 64, 100, 255, 256, 1000, 4095, 4096, 70_000, 200_000, 300_000] {
            let input: Vec<u8> = (0..n)
                .map(|i| {
                    s ^= s << 13;
                    s ^= s >> 17;
                    s ^= s << 5;
                    if (i / 64) % 3 == 0 { (s >> 16) as u8 } else { b'a' + (i % 7) as u8 }
                })
                .collect();
            for level in [1, 3, 6, 9] {
                roundtrip(&input, level);
            }
        }
    }
}
