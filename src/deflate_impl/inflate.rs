//! DEFLATE decompression (RFC 1951)

use std::io;
use std::sync::OnceLock;

use super::bitreader::BitReader;
use super::huffman::{self, Alphabet, HuffmanDecoder};
use super::tables;

const LITLEN_TABLE_BITS: u32 = 10;
const DIST_TABLE_BITS: u32 = 8;
const PRECODE_TABLE_BITS: u32 = 7;

/// Two literals + the longest match (258) + the wildcopy overshoot of the
/// match kernel — what one loop iteration may write past `op`.
const OUT_HEADROOM: usize = 2 + 258 + 16;

/// Decompress a raw DEFLATE stream, appending to `output`.
/// Returns the number of input bytes consumed.
#[cfg_attr(not(test), allow(dead_code))]
pub fn inflate_into(input: &[u8], output: &mut Vec<u8>) -> io::Result<usize> {
    inflate_streaming(input, output, None, &mut |_| {}).map(|(n, _)| n)
}

/// [`inflate_into`] with optional streaming output: with a sink, `output`
/// is a scratch buffer flushed every few MB down to the 32 KiB window (see
/// `crate::Streamer`); `on_flush` sees every flushed byte once (for the
/// gzip CRC / zlib Adler-32). Returns `(input consumed, output produced)`;
/// the tail is left in `output` for the caller to hash and flush.
pub fn inflate_streaming(
    input: &[u8],
    output: &mut Vec<u8>,
    sink: Option<&mut dyn std::io::Write>,
    on_flush: &mut dyn FnMut(&[u8]),
) -> io::Result<(usize, usize)> {
    let mut reader = BitReader::new(input);
    let mut st = crate::Streamer::new(sink, 32 << 10, 4 << 20);
    let out_start = output.len();

    loop {
        let bfinal = reader.read_bits(1)?;
        let btype = reader.read_bits(2)?;

        match btype {
            0 => inflate_stored(&mut reader, output)?,
            1 => inflate_fixed(&mut reader, output)?,
            2 => inflate_dynamic(&mut reader, output)?,
            _ => {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidData,
                    "invalid deflate block type 3",
                ))
            }
        }

        if bfinal == 1 {
            break;
        }
        st.maybe_flush(output, &mut *on_flush)?;
    }

    Ok((reader.bytes_consumed(), st.flushed + output.len() - out_start))
}

// ---------------------------------------------------------------------------
// Block type 0: stored (no compression)
// ---------------------------------------------------------------------------

fn inflate_stored(reader: &mut BitReader, output: &mut Vec<u8>) -> io::Result<()> {
    let header = reader.read_bytes(4)?;
    let len = u16::from_le_bytes([header[0], header[1]]);
    let nlen = u16::from_le_bytes([header[2], header[3]]);

    if len != !nlen {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "stored block LEN/NLEN mismatch",
        ));
    }

    let data = reader.read_bytes(len as usize)?;
    output.extend_from_slice(data);
    Ok(())
}

// ---------------------------------------------------------------------------
// Block type 1: fixed Huffman codes
// ---------------------------------------------------------------------------

fn fixed_tables() -> &'static (HuffmanDecoder, HuffmanDecoder) {
    static TABLES: OnceLock<(HuffmanDecoder, HuffmanDecoder)> = OnceLock::new();
    TABLES.get_or_init(|| {
        (
            HuffmanDecoder::from_lengths(&tables::fixed_literal_lengths(), Alphabet::LitLen, LITLEN_TABLE_BITS)
                .expect("fixed literal table"),
            HuffmanDecoder::from_lengths(&tables::fixed_distance_lengths(), Alphabet::Distance, DIST_TABLE_BITS)
                .expect("fixed distance table"),
        )
    })
}

fn inflate_fixed(reader: &mut BitReader, output: &mut Vec<u8>) -> io::Result<()> {
    let (lit_dec, dist_dec) = fixed_tables();
    decode_block(reader, lit_dec, dist_dec, output)
}

// ---------------------------------------------------------------------------
// Block type 2: dynamic Huffman codes
// ---------------------------------------------------------------------------

fn inflate_dynamic(reader: &mut BitReader, output: &mut Vec<u8>) -> io::Result<()> {
    let hlit = reader.read_bits(5)? as usize + 257;
    let hdist = reader.read_bits(5)? as usize + 1;
    let hclen = reader.read_bits(4)? as usize + 4;

    let mut cl_lengths = [0u8; 19];
    for i in 0..hclen {
        cl_lengths[tables::CODE_LENGTH_ORDER[i]] = reader.read_bits(3)? as u8;
    }
    let cl_dec = HuffmanDecoder::from_lengths(&cl_lengths, Alphabet::Precode, PRECODE_TABLE_BITS)?;

    let total = hlit + hdist;
    let mut all_lengths: Vec<u8> = Vec::with_capacity(total);

    while all_lengths.len() < total {
        let sym = cl_dec.decode(reader)?;
        match sym {
            0..=15 => all_lengths.push(sym as u8),
            16 => {
                let count = reader.read_bits(2)? as usize + 3;
                let prev = *all_lengths
                    .last()
                    .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidData, "code 16 with no previous length"))?;
                let count = count.min(total - all_lengths.len());
                for _ in 0..count {
                    all_lengths.push(prev);
                }
            }
            17 => {
                let count = reader.read_bits(3)? as usize + 3;
                let count = count.min(total - all_lengths.len());
                for _ in 0..count {
                    all_lengths.push(0);
                }
            }
            18 => {
                let count = reader.read_bits(7)? as usize + 11;
                let count = count.min(total - all_lengths.len());
                for _ in 0..count {
                    all_lengths.push(0);
                }
            }
            _ => {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidData,
                    "invalid code-length symbol",
                ))
            }
        }
    }

    let lit_dec = HuffmanDecoder::from_lengths(&all_lengths[..hlit], Alphabet::LitLen, LITLEN_TABLE_BITS)?;
    let dist_dec = HuffmanDecoder::from_lengths(&all_lengths[hlit..], Alphabet::Distance, DIST_TABLE_BITS)?;
    decode_block(reader, &lit_dec, &dist_dec, output)
}

// ---------------------------------------------------------------------------
// Decode compressed data within a block
// ---------------------------------------------------------------------------

#[cold]
#[inline(never)]
fn err(kind: io::ErrorKind, msg: &'static str) -> io::Error {
    io::Error::new(kind, msg)
}

/// Raw-pointer output cursor over a `Vec<u8>` with a guaranteed headroom of
/// `OUT_HEADROOM` bytes past `op` while decoding (grown on demand).
struct OutCursor<'a> {
    vec: &'a mut Vec<u8>,
    base: *mut u8,
    op: *mut u8,
    end: *mut u8,
}

impl<'a> OutCursor<'a> {
    fn new(vec: &'a mut Vec<u8>) -> Self {
        let mut c = OutCursor { vec, base: std::ptr::null_mut(), op: std::ptr::null_mut(), end: std::ptr::null_mut() };
        c.grow();
        c
    }

    #[cold]
    #[inline(never)]
    fn grow(&mut self) {
        let len = if self.base.is_null() { self.vec.len() } else { self.op as usize - self.base as usize };
        // SAFETY: `len` bytes have been written below `op`.
        unsafe { self.vec.set_len(len) };
        self.vec.reserve((64 * 1024).max(OUT_HEADROOM));
        self.base = self.vec.as_mut_ptr();
        // SAFETY: `len <= capacity`, so both offsets stay within (or one
        // past the end of) the Vec's allocation.
        self.op = unsafe { self.base.add(len) };
        // SAFETY: as above.
        self.end = unsafe { self.base.add(self.vec.capacity()) };
    }

    #[inline(always)]
    fn ensure_headroom(&mut self) {
        if (self.end as usize - self.op as usize) < OUT_HEADROOM {
            self.grow();
        }
    }

    fn finish(self) {
        let len = self.op as usize - self.base as usize;
        // SAFETY: every byte below `op` was written.
        unsafe { self.vec.set_len(len) };
    }
}

fn decode_block(
    reader: &mut BitReader,
    lit_dec: &HuffmanDecoder,
    dist_dec: &HuffmanDecoder,
    output: &mut Vec<u8>,
) -> io::Result<()> {
    use huffman::{EXTRA_MASK, EXTRA_SHIFT, KIND_BASE, KIND_LITERAL, KIND_MASK, KIND_SHIFT, LEN_MASK};

    let mut out = OutCursor::new(output);

    loop {
        // One refill covers a whole sequence: litlen code (15) + length extra
        // (5) + distance code (15) + distance extra (13) = 48 <= 56 bits.
        if !reader.refill() {
            return Err(err(io::ErrorKind::UnexpectedEof, "unexpected end of deflate stream"));
        }
        out.ensure_headroom();

        let mut e = lit_dec.lookup::<LITLEN_TABLE_BITS>(reader);
        reader.consume(e & LEN_MASK);
        if e & KIND_MASK == KIND_LITERAL << KIND_SHIFT {
            // SAFETY: headroom guaranteed above (covers 2 literals + a match).
            unsafe {
                *out.op = (e >> 16) as u8;
                out.op = out.op.add(1);
            }
            // A second literal fits in the same refill (>= 41 bits left).
            e = lit_dec.lookup::<LITLEN_TABLE_BITS>(reader);
            reader.consume(e & LEN_MASK);
            if e & KIND_MASK == KIND_LITERAL << KIND_SHIFT {
                // SAFETY: same headroom as the first literal above.
                unsafe {
                    *out.op = (e >> 16) as u8;
                    out.op = out.op.add(1);
                }
                continue;
            }
            // Length (or special): top up for the rest of the sequence.
            if !reader.refill() {
                out.finish();
                return Err(err(io::ErrorKind::UnexpectedEof, "unexpected end of deflate stream"));
            }
        }
        let kind = e & KIND_MASK;
        if kind != KIND_BASE << KIND_SHIFT {
            // Special: end-of-block or invalid.
            out.finish();
            if reader.overread() {
                return Err(err(io::ErrorKind::UnexpectedEof, "unexpected end of deflate stream"));
            }
            if (e >> 16) == huffman::EOB_ENTRY_VALUE {
                return Ok(());
            }
            return Err(err(io::ErrorKind::InvalidData, "invalid literal/length symbol"));
        }

        let length = (e >> 16) as usize + reader.bits((e >> EXTRA_SHIFT) & EXTRA_MASK) as usize;

        let d = dist_dec.lookup::<DIST_TABLE_BITS>(reader);
        reader.consume(d & LEN_MASK);
        if d & KIND_MASK != KIND_BASE << KIND_SHIFT {
            out.finish();
            return Err(err(io::ErrorKind::InvalidData, "invalid distance code"));
        }
        let distance = (d >> 16) as usize + reader.bits((d >> EXTRA_SHIFT) & EXTRA_MASK) as usize;

        let produced = out.op as usize - out.base as usize;
        if distance > produced {
            out.finish();
            return Err(err(io::ErrorKind::InvalidData, "distance too far back"));
        }
        if reader.overread() {
            out.finish();
            return Err(err(io::ErrorKind::UnexpectedEof, "unexpected end of deflate stream"));
        }
        // SAFETY: `distance <= produced` so the source is inside the written
        // region; `OUT_HEADROOM` guarantees `length + 16` bytes past `op`.
        unsafe {
            crate::cpu_features::copy_match_unchecked(out.op.sub(distance), out.op, distance, length);
            out.op = out.op.add(length);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    #[test]
    fn test_inflate_flate2_level1_text() {
        let text = b"The quick brown fox jumps over the lazy dog. ".repeat(200);
        let mut enc = flate2::write::DeflateEncoder::new(Vec::new(), flate2::Compression::new(1));
        enc.write_all(&text).unwrap();
        let compressed = enc.finish().unwrap();

        let mut out = Vec::new();
        let consumed = inflate_into(&compressed, &mut out).unwrap();
        assert_eq!(consumed, compressed.len());
        assert_eq!(out, text);
    }

    #[test]
    // Every level over a large input: > 8 min under Miri; the small
    // trailing-garbage and level tests cover the same paths.
    #[cfg_attr(miri, ignore)]
    fn test_inflate_all_levels_with_trailing_garbage() {
        let mut data = Vec::new();
        for i in 0..50_000u32 {
            data.push((i.wrapping_mul(2654435761) >> 13) as u8 & 0x1F | 0x40);
        }
        for level in 0..=9 {
            let mut enc = flate2::write::DeflateEncoder::new(Vec::new(), flate2::Compression::new(level));
            enc.write_all(&data).unwrap();
            let mut compressed = enc.finish().unwrap();
            let clen = compressed.len();
            compressed.extend_from_slice(&[0xAA; 20]);
            let mut out = Vec::new();
            let consumed = inflate_into(&compressed, &mut out).unwrap();
            assert_eq!(consumed, clen, "level {level}");
            assert_eq!(out, data, "level {level}");
        }
    }

    #[test]
    fn test_inflate_truncated_is_error() {
        let text = b"hello hello hello hello hello world".repeat(100);
        let mut enc = flate2::write::DeflateEncoder::new(Vec::new(), flate2::Compression::new(6));
        enc.write_all(&text).unwrap();
        let compressed = enc.finish().unwrap();
        for cut in [1usize, 5, compressed.len() / 2, compressed.len() - 1] {
            let mut out = Vec::new();
            assert!(inflate_into(&compressed[..cut], &mut out).is_err(), "cut at {cut}");
        }
    }
}
