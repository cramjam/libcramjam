//! LZ4 frame format encoder/decoder.
//!
//! Spec: <https://github.com/lz4/lz4/blob/dev/doc/lz4_Frame_format.md>
//!
//! Frame layout:
//!   * Magic Number (4 bytes LE) = 0x184D2204
//!   * Frame Descriptor (3..15 bytes): FLG byte, BD byte, optional content
//!     size (8 bytes), optional dict ID (4 bytes), 1-byte header checksum
//!     (xxhash32 of FLG..end-of-descriptor, byte 2 of the result).
//!   * Series of data blocks:
//!     - Block size (4 bytes LE).  High bit set ⇒ uncompressed.  Size 0 ⇒ end.
//!     - Block data (compressed or raw).
//!     - Optional 4-byte block xxhash32 checksum (if BD enables it).
//!   * End-of-frame marker = 4 zero bytes.
//!   * Optional 4-byte content xxhash32 checksum (if FLG enables it).

use std::io;

use super::block;

const LZ4_FRAME_MAGIC: u32 = 0x184D2204;
const FLG_VERSION_BITS: u8 = 0b0100_0000; // version = 01

/// Default 64 KiB block size code.  Block sizes per the spec:
///   4 → 64 KiB
///   5 → 256 KiB
///   6 → 1 MiB
///   7 → 4 MiB
const DEFAULT_BLOCK_SIZE_CODE: u8 = 4;
const DEFAULT_BLOCK_SIZE: usize = 64 * 1024;

/// Sentinel high-bit for the "uncompressed block" flag.
const UNCOMPRESSED_BIT: u32 = 1 << 31;

// =========================================================================
// Encoder
// =========================================================================

#[cfg(test)]
pub fn encode_frame(input: &[u8]) -> Vec<u8> {
    encode_frame_at(input, None)
}

/// Encode an LZ4 frame at the requested level.
///
/// `level == None` or `Some(0..=1)` uses the fast parser
/// (`block::compress_block_fast_continue`); `Some(2..=12)` selects the HC
/// context (level 2 = `LZ4MID`, 3-9 hash chain, 10-12 optimal) — matching
/// lz4frame's `LZ4HC_CLEVEL_MIN` routing.
#[cfg(test)]
pub fn encode_frame_at(input: &[u8], level: Option<u32>) -> Vec<u8> {
    encode_frame_opts(input, level, true, false)
}

/// FLG bit 5: blocks are independent (no back-references across blocks).
const FLG_BLOCK_INDEPENDENT: u8 = 1 << 5;
/// FLG bit 2: a 4-byte xxhash32 of the content follows the end marker.
const FLG_CONTENT_CHECKSUM: u8 = 1 << 2;

/// [`encode_frame_at`] with the frame options the C `LZ4F_preferences_t`
/// exposes: `block_linked = false` resets the match-finder context at every
/// block (`LZ4F_blockIndependent`), `content_checksum` appends the xxhash32
/// of the input after the end marker.
pub fn encode_frame_opts(input: &[u8], level: Option<u32>, block_linked: bool, content_checksum: bool) -> Vec<u8> {
    let mut out = Vec::with_capacity(input.len() + 32);
    encode_frame_opts_into(&mut out, input, level, block_linked, content_checksum);
    out
}

/// [`encode_frame_opts`] appending to a caller-provided buffer.
pub fn encode_frame_opts_into(out: &mut Vec<u8>, input: &[u8], level: Option<u32>, block_linked: bool, content_checksum: bool) {
    write_frame_header(out, block_linked, content_checksum);
    // `LZ4HC_CLEVEL_MIN` = 2: C routes level 2 to the HC context (LZ4MID).
    let use_hc = use_hc_for(level);

    let hc_level = level.unwrap_or(0);
    let mut hc_ctx = if use_hc { Some(super::hc::HcCtx::new()) } else { None };
    let mut fast_ctx = if use_hc { None } else { Some(block::FastCtx::new()) };

    // Blocks.
    let mut pos = 0;
    let mut compressed = Vec::with_capacity(block::compress_bound(DEFAULT_BLOCK_SIZE));
    while pos < input.len() {
        let chunk_end = (pos + DEFAULT_BLOCK_SIZE).min(input.len());
        let chunk = &input[pos..chunk_end];

        // Compress the block; if the compressed payload >= raw size, emit raw.
        compressed.clear();
        if !block_linked {
            compress_independent_block(chunk, hc_level, use_hc, &mut compressed);
        } else if let Some(ctx) = hc_ctx.as_mut() {
            super::hc::compress_block_hc_continue(ctx, input, pos, chunk_end, &mut compressed, hc_level);
        } else {
            block::compress_block_fast_continue(fast_ctx.as_mut().unwrap(), input, pos, chunk_end, &mut compressed);
        }

        emit_block(out, &compressed, chunk);
        pos = chunk_end;
    }

    // End-of-frame marker.
    out.extend_from_slice(&0u32.to_le_bytes());
    if content_checksum {
        out.extend_from_slice(&xxhash32(input, 0).to_le_bytes());
    }
}

/// One independent block (`LZ4F_blockIndependent`): the block is its own
/// input, so nothing — not even a backward match extension — can reach
/// before its first byte.
pub(crate) fn compress_independent_block(chunk: &[u8], level: u32, use_hc: bool, out: &mut Vec<u8>) {
    if use_hc {
        let mut ctx = super::hc::HcCtx::new();
        super::hc::compress_block_hc_continue(&mut ctx, chunk, 0, chunk.len(), out, level);
    } else {
        block::compress_block(chunk, out);
    }
}

pub(crate) fn use_hc_for(level: Option<u32>) -> bool {
    matches!(level, Some(l) if l >= 2)
}

/// Magic + frame descriptor (FLG, BD = 64 KiB blocks, header checksum).
pub(crate) fn write_frame_header(out: &mut Vec<u8>, block_linked: bool, content_checksum: bool) {
    out.extend_from_slice(&LZ4_FRAME_MAGIC.to_le_bytes());
    let mut flg: u8 = FLG_VERSION_BITS;
    if !block_linked {
        flg |= FLG_BLOCK_INDEPENDENT;
    }
    if content_checksum {
        flg |= FLG_CONTENT_CHECKSUM;
    }
    let bd: u8 = DEFAULT_BLOCK_SIZE_CODE << 4;
    let start = out.len();
    out.push(flg);
    out.push(bd);
    // Header checksum: byte 2 of xxhash32 over FLG..BD.
    let hc = xxhash32(&out[start..start + 2], 0);
    out.push(((hc >> 8) & 0xFF) as u8);
}

/// One block: the compressed payload if it shrank, else the raw chunk with
/// the uncompressed bit set.
pub(crate) fn emit_block(out: &mut Vec<u8>, compressed: &[u8], chunk: &[u8]) {
    if compressed.len() < chunk.len() {
        let size = compressed.len() as u32;
        out.extend_from_slice(&size.to_le_bytes());
        out.extend_from_slice(compressed);
    } else {
        let size = (chunk.len() as u32) | UNCOMPRESSED_BIT;
        out.extend_from_slice(&size.to_le_bytes());
        out.extend_from_slice(chunk);
    }
}

/// Incremental xxhash32 (seed 0) for the frame content checksum.
pub(crate) struct Xxh32 {
    v: [u32; 4],
    buf: [u8; 16],
    buf_len: usize,
    total: u64,
}

impl Xxh32 {
    pub fn new() -> Self {
        Self {
            v: [PRIME32_1.wrapping_add(PRIME32_2), PRIME32_2, 0, 0u32.wrapping_sub(PRIME32_1)],
            buf: [0; 16],
            buf_len: 0,
            total: 0,
        }
    }

    fn stripe(&mut self, b: &[u8]) {
        for k in 0..4 {
            self.v[k] = round32(self.v[k], read_u32_le(b, k * 4));
        }
    }

    pub fn update(&mut self, mut data: &[u8]) {
        self.total += data.len() as u64;
        if self.buf_len > 0 {
            let take = (16 - self.buf_len).min(data.len());
            self.buf[self.buf_len..self.buf_len + take].copy_from_slice(&data[..take]);
            self.buf_len += take;
            data = &data[take..];
            if self.buf_len < 16 {
                return;
            }
            let b = self.buf;
            self.stripe(&b);
            self.buf_len = 0;
        }
        let mut i = 0;
        while i + 16 <= data.len() {
            self.stripe(&data[i..i + 16]);
            i += 16;
        }
        let rest = data.len() - i;
        self.buf[..rest].copy_from_slice(&data[i..]);
        self.buf_len = rest;
    }

    pub fn finish(&self) -> u32 {
        let mut h = if self.total >= 16 {
            self.v[0].rotate_left(1)
                .wrapping_add(self.v[1].rotate_left(7))
                .wrapping_add(self.v[2].rotate_left(12))
                .wrapping_add(self.v[3].rotate_left(18))
        } else {
            PRIME32_5
        };
        h = h.wrapping_add(self.total as u32);
        let data = &self.buf[..self.buf_len];
        let mut i = 0;
        while i + 4 <= data.len() {
            h = h.wrapping_add(read_u32_le(data, i).wrapping_mul(PRIME32_3));
            h = h.rotate_left(17).wrapping_mul(PRIME32_4);
            i += 4;
        }
        while i < data.len() {
            h = h.wrapping_add((data[i] as u32).wrapping_mul(PRIME32_5));
            h = h.rotate_left(11).wrapping_mul(PRIME32_1);
            i += 1;
        }
        h ^= h >> 15;
        h = h.wrapping_mul(PRIME32_2);
        h ^= h >> 13;
        h = h.wrapping_mul(PRIME32_3);
        h ^= h >> 16;
        h
    }
}

// =========================================================================
// Decoder
// =========================================================================

#[cfg_attr(not(test), allow(dead_code))]
pub fn decode_frame(input: &[u8], output: &mut Vec<u8>) -> io::Result<usize> {
    decode_frame_streaming(input, output, None).map(|(n, _)| n)
}

/// [`decode_frame`] with optional streaming output: with a sink, `output`
/// is a scratch buffer that is flushed every few MB down to the 64 KiB
/// block-dependency window (see `crate::Streamer`). Returns
/// `(input consumed, output produced)`; the tail is left in `output`.
pub fn decode_frame_streaming(
    input: &[u8],
    output: &mut Vec<u8>,
    sink: Option<&mut dyn std::io::Write>,
) -> io::Result<(usize, usize)> {
    let streaming = sink.is_some();
    let mut st = crate::Streamer::new(sink, 64 << 10, 4 << 20);
    if input.len() < 7 {
        return Err(io::Error::new(io::ErrorKind::UnexpectedEof, "lz4: frame too short"));
    }
    let magic = u32::from_le_bytes([input[0], input[1], input[2], input[3]]);
    if magic != LZ4_FRAME_MAGIC {
        // Skippable frames have magic 0x184D2A50..0x184D2A5F.
        if (magic & 0xFFFFFFF0) == 0x184D2A50 {
            if input.len() < 8 {
                return Err(io::Error::new(io::ErrorKind::UnexpectedEof, "lz4: skippable frame truncated"));
            }
            let size = u32::from_le_bytes([input[4], input[5], input[6], input[7]]) as usize;
            return Ok((8 + size, 0));
        }
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!("lz4: bad magic 0x{:08x}", magic),
        ));
    }

    let mut p = 4usize;
    let flg = input[p];
    p += 1;
    let bd = input[p];
    p += 1;

    let version = (flg >> 6) & 3;
    if version != 1 {
        return Err(io::Error::new(io::ErrorKind::InvalidData, "lz4: unsupported frame version"));
    }
    let block_indep = (flg & 0x20) != 0;
    let block_checksum = (flg & 0x10) != 0;
    let content_size_present = (flg & 0x08) != 0;
    let content_checksum = (flg & 0x04) != 0;
    let dict_id_present = (flg & 0x01) != 0;

    let _ = block_indep;
    // Block maximum size (BD byte, bits 4..6): 4 → 64 KiB … 7 → 4 MiB.
    // Anything else is invalid per spec; treat it as 4 MiB for sizing only.
    let block_max: usize = match (bd >> 4) & 7 {
        4 => 64 << 10,
        5 => 256 << 10,
        6 => 1 << 20,
        _ => 4 << 20,
    };

    let mut content_size: Option<u64> = None;
    if content_size_present {
        if p + 8 > input.len() {
            return Err(io::Error::new(io::ErrorKind::UnexpectedEof, "lz4: content size truncated"));
        }
        content_size = Some(u64::from_le_bytes([
            input[p], input[p + 1], input[p + 2], input[p + 3],
            input[p + 4], input[p + 5], input[p + 6], input[p + 7],
        ]));
        p += 8;
    }
    if dict_id_present {
        if p + 4 > input.len() {
            return Err(io::Error::new(io::ErrorKind::UnexpectedEof, "lz4: dict id truncated"));
        }
        p += 4;
    }
    // Header checksum byte (we don't validate it — non-critical).
    if p >= input.len() {
        return Err(io::Error::new(io::ErrorKind::UnexpectedEof, "lz4: header checksum truncated"));
    }
    p += 1;

    if streaming {
        output.clear();
    }
    let out_start = output.len();
    // Size the output once so the block decoder never has to grow it:
    // exact when the frame carries its content size, else amortised via
    // one block-max step per block.
    if let Some(cs) = content_size {
        let cs = cs.min(1 << 32) as usize;
        output.reserve(if streaming { cs.min(8 << 20) } else { cs } + block::OUT_SLACK);
    }

    // Blocks.
    loop {
        if p + 4 > input.len() {
            return Err(io::Error::new(io::ErrorKind::UnexpectedEof, "lz4: block size truncated"));
        }
        let block_size_word = u32::from_le_bytes([input[p], input[p + 1], input[p + 2], input[p + 3]]);
        p += 4;

        if block_size_word == 0 {
            // End-of-frame marker.
            break;
        }

        let uncompressed = (block_size_word & UNCOMPRESSED_BIT) != 0;
        let block_size = (block_size_word & !UNCOMPRESSED_BIT) as usize;
        if p + block_size > input.len() {
            return Err(io::Error::new(io::ErrorKind::UnexpectedEof, "lz4: block data truncated"));
        }
        let block_data = &input[p..p + block_size];
        p += block_size;

        if uncompressed {
            output.extend_from_slice(block_data);
        } else {
            output.reserve(block_max + block::OUT_SLACK);
            block::decompress_block(block_data, output)?;
        }

        if block_checksum {
            if p + 4 > input.len() {
                return Err(io::Error::new(io::ErrorKind::UnexpectedEof, "lz4: block checksum truncated"));
            }
            // We don't verify the per-block checksum (non-critical).
            p += 4;
        }
        st.maybe_flush(output, |_| {})?;
    }

    if content_checksum {
        if p + 4 > input.len() {
            return Err(io::Error::new(io::ErrorKind::UnexpectedEof, "lz4: content checksum truncated"));
        }
        // We don't verify the content checksum (non-critical).
        p += 4;
    }

    let produced = st.flushed + output.len() - out_start;
    if let Some(cs) = content_size {
        if produced as u64 != cs {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!("lz4: content size mismatch (header={}, produced={})", cs, produced),
            ));
        }
    }

    Ok((p, produced))
}

// =========================================================================
// xxhash32 (used for the header checksum byte; we don't verify it on decode
// but we do produce a correct one on encode).
// =========================================================================

const PRIME32_1: u32 = 0x9E3779B1;
const PRIME32_2: u32 = 0x85EBCA77;
const PRIME32_3: u32 = 0xC2B2AE3D;
const PRIME32_4: u32 = 0x27D4EB2F;
const PRIME32_5: u32 = 0x165667B1;

fn xxhash32(data: &[u8], seed: u32) -> u32 {
    let len = data.len();
    let mut h: u32;
    let mut i = 0usize;

    if len >= 16 {
        let mut v1 = seed.wrapping_add(PRIME32_1).wrapping_add(PRIME32_2);
        let mut v2 = seed.wrapping_add(PRIME32_2);
        let mut v3 = seed;
        let mut v4 = seed.wrapping_sub(PRIME32_1);

        while i + 16 <= len {
            v1 = round32(v1, read_u32_le(data, i));
            v2 = round32(v2, read_u32_le(data, i + 4));
            v3 = round32(v3, read_u32_le(data, i + 8));
            v4 = round32(v4, read_u32_le(data, i + 12));
            i += 16;
        }
        h = v1.rotate_left(1)
            .wrapping_add(v2.rotate_left(7))
            .wrapping_add(v3.rotate_left(12))
            .wrapping_add(v4.rotate_left(18));
    } else {
        h = seed.wrapping_add(PRIME32_5);
    }

    h = h.wrapping_add(len as u32);

    while i + 4 <= len {
        let k = read_u32_le(data, i);
        h = h.wrapping_add(k.wrapping_mul(PRIME32_3));
        h = h.rotate_left(17).wrapping_mul(PRIME32_4);
        i += 4;
    }
    while i < len {
        h = h.wrapping_add((data[i] as u32).wrapping_mul(PRIME32_5));
        h = h.rotate_left(11).wrapping_mul(PRIME32_1);
        i += 1;
    }

    h ^= h >> 15;
    h = h.wrapping_mul(PRIME32_2);
    h ^= h >> 13;
    h = h.wrapping_mul(PRIME32_3);
    h ^= h >> 16;
    h
}

#[inline]
fn round32(mut acc: u32, input: u32) -> u32 {
    acc = acc.wrapping_add(input.wrapping_mul(PRIME32_2));
    acc = acc.rotate_left(13);
    acc.wrapping_mul(PRIME32_1)
}

#[inline]
fn read_u32_le(data: &[u8], pos: usize) -> u32 {
    u32::from_le_bytes([data[pos], data[pos + 1], data[pos + 2], data[pos + 3]])
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn frame_roundtrip_text() {
        let input: Vec<u8> = b"The quick brown fox jumps over the lazy dog. ".repeat(100);
        let frame = encode_frame(&input);
        let mut decoded = Vec::new();
        decode_frame(&frame, &mut decoded).unwrap();
        assert_eq!(decoded, input);
    }

    #[test]
    fn frame_roundtrip_empty() {
        let frame = encode_frame(&[]);
        let mut decoded = Vec::new();
        decode_frame(&frame, &mut decoded).unwrap();
        assert!(decoded.is_empty());
    }

    #[test]
    fn frame_roundtrip_large() {
        let input: Vec<u8> = (0..200_000u32).map(|i| (i * 31 + 7) as u8).collect();
        let frame = encode_frame(&input);
        let mut decoded = Vec::new();
        decode_frame(&frame, &mut decoded).unwrap();
        assert_eq!(decoded, input);
    }
}
