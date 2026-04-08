//! LZ4 block format encoder/decoder.
//!
//! The block format is described at
//! <https://github.com/lz4/lz4/blob/dev/doc/lz4_Block_format.md>.
//!
//! A block is a sequence of *sequences*, each of which contains a literal run
//! followed by a back-reference (match).  The last sequence has only literals.
//!
//! Sequence layout:
//! ```text
//!   token (1 byte) |  literal_len_bytes...  |  literals  |
//!   offset (2 bytes LE)                                  |
//!   match_len_bytes...                                   |
//! ```
//!
//! The token packs `literal_length` (high nibble) and `match_length - 4`
//! (low nibble).  When either value is 15, additional length bytes follow
//! (one or more 255 + final byte 0..254 summed).
//!
//! Constraints (all enforced):
//!   * Min match = 4 bytes.
//!   * Min offset = 1, max offset = 65 535.
//!   * The last 5 bytes of any block are always literals.
//!   * The last match must end at least 12 bytes before the end of the block.

use std::io;

const MIN_MATCH: usize = 4;
const MAX_OFFSET: usize = 65_535;
/// Lookback distance for the LAST possible match — last 12 bytes of any
/// block must be literals (RFC requirement).
const MIN_TRAILING_LITERALS: usize = 12;
const HASH_BITS: usize = 14;
const HASH_SIZE: usize = 1 << HASH_BITS;
const HASH_MASK: usize = HASH_SIZE - 1;
const NONE: u32 = u32::MAX;

// =========================================================================
// Decoder
// =========================================================================

/// Decompress an LZ4 block into `output`.  Returns the number of OUTPUT bytes
/// written.  Reserves an aggressive amount of capacity up-front so the inner
/// loop can avoid `Vec` growth checks via direct unchecked writes.
pub fn decompress_block(input: &[u8], output: &mut Vec<u8>) -> io::Result<usize> {
    // Reserve a generous amount of headroom so we never reallocate inside the
    // hot loop.  Worst case is bounded by `input.len() * 256` (single token,
    // 1 byte literal + 65535-byte match), but we cap at the LZ4 frame block
    // limit + slack.
    let upper_bound = (input.len() * 256 + 65_536).min(8 * 1024 * 1024);
    output.reserve(upper_bound);

    let start = output.len();
    let mut ip = 0usize;
    let in_len = input.len();

    while ip < in_len {
        let token = unsafe { *input.get_unchecked(ip) };
        ip += 1;

        // -- Literal run --
        let mut lit_len = (token >> 4) as usize;
        if lit_len == 15 {
            loop {
                if ip >= in_len {
                    return Err(io::Error::new(
                        io::ErrorKind::UnexpectedEof,
                        "lz4: unexpected end while reading literal length",
                    ));
                }
                let b = unsafe { *input.get_unchecked(ip) };
                ip += 1;
                lit_len += b as usize;
                if b != 255 {
                    break;
                }
            }
        }

        if ip + lit_len > in_len {
            return Err(io::Error::new(
                io::ErrorKind::UnexpectedEof,
                "lz4: literal run exceeds input",
            ));
        }
        // Wildcopy literals: copy in 8-byte chunks then trim.  We reserved
        // headroom above so writing past the current len is in-bounds.
        copy_literals(output, &input[ip..ip + lit_len]);
        ip += lit_len;

        // End of block: when there are no more bytes, we're done (the last
        // sequence has only literals, no match field).
        if ip == in_len {
            break;
        }

        // -- Match --
        if ip + 2 > in_len {
            return Err(io::Error::new(
                io::ErrorKind::UnexpectedEof,
                "lz4: missing match offset",
            ));
        }
        let offset = u16::from_le_bytes([
            unsafe { *input.get_unchecked(ip) },
            unsafe { *input.get_unchecked(ip + 1) },
        ]) as usize;
        ip += 2;
        if offset == 0 {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "lz4: zero match offset",
            ));
        }

        let mut match_len = (token & 0x0F) as usize;
        if match_len == 15 {
            loop {
                if ip >= in_len {
                    return Err(io::Error::new(
                        io::ErrorKind::UnexpectedEof,
                        "lz4: unexpected end while reading match length",
                    ));
                }
                let b = unsafe { *input.get_unchecked(ip) };
                ip += 1;
                match_len += b as usize;
                if b != 255 {
                    break;
                }
            }
        }
        match_len += MIN_MATCH;

        let cur = output.len();
        if offset > cur {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "lz4: match offset beyond output",
            ));
        }
        copy_match(output, offset, match_len);
    }

    Ok(output.len() - start)
}

/// Copy a literal run into `output`.  Uses `extend_from_slice` which calls
/// memcpy internally; the up-front `reserve` in the caller eliminates the
/// per-call growth check.
#[inline(always)]
fn copy_literals(output: &mut Vec<u8>, literals: &[u8]) {
    output.extend_from_slice(literals);
}

/// Copy a match (back-reference) into `output` using LZ4's wildcopy strategy.
/// Caller must have reserved enough capacity in `output`.
#[inline(always)]
fn copy_match(output: &mut Vec<u8>, offset: usize, match_len: usize) {
    let cur = output.len();
    let match_start = cur - offset;
    let end = cur + match_len;
    debug_assert!(output.capacity() >= end + 16);

    if offset >= 8 && end + 16 <= output.capacity() {
        // Wildcopy 8-byte chunks.  Reading from src is always safe because
        // the bytes we read in chunk N were written by an earlier chunk
        // (or already exist as initialized output).  The dst writes go into
        // reserved spare capacity, which we then truncate via set_len.
        unsafe {
            let base = output.as_mut_ptr();
            let mut i = 0usize;
            while i < match_len {
                let src = base.add(match_start + i);
                let dst = base.add(cur + i);
                std::ptr::copy_nonoverlapping(src, dst, 8);
                i += 8;
            }
            output.set_len(end);
        }
    } else {
        // Slow / overlapping path: 1 byte at a time.  Required when
        // offset < 8 (small-period RLE) OR when we're near the end of the
        // reserved capacity and can't safely overshoot by 8 bytes.
        for i in 0..match_len {
            let b = output[match_start + i];
            output.push(b);
        }
    }
}

// =========================================================================
// Encoder
// =========================================================================

/// Worst-case compressed size for an `n`-byte input — used to size output buffers.
/// Matches LZ4_compressBound.
pub fn compress_bound(n: usize) -> usize {
    n + n / 255 + 16
}

/// Compress `input` as an LZ4 block.  Always falls back to a single
/// literal-only sequence if the input is too small for any match.
pub fn compress_block(input: &[u8], output: &mut Vec<u8>) -> usize {
    let start_out = output.len();

    if input.len() < MIN_TRAILING_LITERALS + MIN_MATCH {
        emit_literal_only(output, input);
        return output.len() - start_out;
    }

    let mut head = vec![NONE; HASH_SIZE];
    let mut ip = 0usize;
    let mut anchor = 0usize;
    let len = input.len();
    let last_match_pos = len - MIN_TRAILING_LITERALS;

    while ip < last_match_pos {
        let h = hash4_lz4(&input[ip..]);
        let cand = head[h];
        head[h] = ip as u32;

        if cand == NONE {
            ip += 1;
            continue;
        }
        let mp = cand as usize;
        let dist = ip - mp;
        if dist == 0 || dist > MAX_OFFSET {
            ip += 1;
            continue;
        }

        // 4-byte prefix check.
        if input[mp] != input[ip]
            || input[mp + 1] != input[ip + 1]
            || input[mp + 2] != input[ip + 2]
            || input[mp + 3] != input[ip + 3]
        {
            ip += 1;
            continue;
        }

        // Found a 4-byte (or longer) match.  Extend forward.
        let mut mlen = MIN_MATCH;
        let max_extend = last_match_pos - ip;
        while mlen < max_extend && input[mp + mlen] == input[ip + mlen] {
            mlen += 1;
        }

        // Emit the sequence (literal run + match).
        let lit_len = ip - anchor;
        emit_sequence(output, &input[anchor..ip], lit_len, dist as u16, mlen);

        ip += mlen;
        anchor = ip;
    }

    // Trailing literals (always at least LAST_LITERALS bytes).
    if anchor < len {
        emit_literal_only(output, &input[anchor..]);
    }

    output.len() - start_out
}

/// Emit a "literal-only" sequence (no match) — used for the trailing data and
/// for inputs too small to compress.
fn emit_literal_only(output: &mut Vec<u8>, literals: &[u8]) {
    let lit_len = literals.len();
    let token_lit_part: u8 = if lit_len < 15 { lit_len as u8 } else { 15 };
    output.push(token_lit_part << 4);
    if lit_len >= 15 {
        let mut remaining = lit_len - 15;
        while remaining >= 255 {
            output.push(255);
            remaining -= 255;
        }
        output.push(remaining as u8);
    }
    output.extend_from_slice(literals);
}

/// Emit a full sequence: literal run + match.
fn emit_sequence(output: &mut Vec<u8>, literals: &[u8], lit_len: usize, offset: u16, match_len: usize) {
    let ml_code = match_len - MIN_MATCH;

    // Token byte: high nibble = lit length code, low nibble = match length code.
    let token_lit: u8 = if lit_len < 15 { lit_len as u8 } else { 15 };
    let token_ml: u8 = if ml_code < 15 { ml_code as u8 } else { 15 };
    output.push((token_lit << 4) | token_ml);

    // Extra literal length bytes.
    if lit_len >= 15 {
        let mut remaining = lit_len - 15;
        while remaining >= 255 {
            output.push(255);
            remaining -= 255;
        }
        output.push(remaining as u8);
    }

    // Literals.
    output.extend_from_slice(literals);

    // Offset (2 bytes LE).
    output.extend_from_slice(&offset.to_le_bytes());

    // Extra match length bytes.
    if ml_code >= 15 {
        let mut remaining = ml_code - 15;
        while remaining >= 255 {
            output.push(255);
            remaining -= 255;
        }
        output.push(remaining as u8);
    }
}

#[inline]
fn hash4_lz4(b: &[u8]) -> usize {
    let v = u32::from_le_bytes([b[0], b[1], b[2], b[3]]);
    (v.wrapping_mul(2654435761) >> (32 - HASH_BITS)) as usize & HASH_MASK
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn block_roundtrip_short() {
        let input = b"hello world";
        let mut compressed = Vec::new();
        compress_block(input, &mut compressed);
        let mut decoded = Vec::new();
        decompress_block(&compressed, &mut decoded).unwrap();
        assert_eq!(decoded.as_slice(), input);
    }

    #[test]
    fn block_roundtrip_long_repeating() {
        let input: Vec<u8> = b"abcdefghijklmnop".repeat(1000);
        let mut compressed = Vec::new();
        compress_block(&input, &mut compressed);
        let mut decoded = Vec::new();
        decompress_block(&compressed, &mut decoded).unwrap();
        assert_eq!(decoded, input);
        // Should compress significantly.
        assert!(compressed.len() < input.len() / 4);
    }

    #[test]
    fn block_roundtrip_random() {
        let mut s: u32 = 0xCAFE_BABE;
        let input: Vec<u8> = (0..8000)
            .map(|_| {
                s ^= s << 13;
                s ^= s >> 17;
                s ^= s << 5;
                (s >> 16) as u8
            })
            .collect();
        let mut compressed = Vec::new();
        compress_block(&input, &mut compressed);
        let mut decoded = Vec::new();
        decompress_block(&compressed, &mut decoded).unwrap();
        assert_eq!(decoded, input);
    }
}
