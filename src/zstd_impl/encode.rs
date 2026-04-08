//! Zstandard frame encoder (RFC 8878).
//!
//! Two paths:
//!   * Raw blocks for level == 0 (no entropy coding) — used as a fallback when
//!     compression fails to make data smaller.
//!   * LZ77 + sequences-with-predefined-FSE-tables for level >= 1.  Literals are
//!     emitted as raw (lit_type = 0) for now.

use super::bits::ForwardBitWriter;
use super::fse::{
    self, FseEncoder, LITLEN_TABLE, MATCHLEN_TABLE,
};

const ZSTD_MAGIC: u32 = 0xFD2FB528;
const MAX_BLOCK_SIZE: usize = 128 * 1024; // 128 KB per RFC

// =========================================================================
// Public entry points
// =========================================================================

/// Compress `input` into a zstd frame.
///
/// `level`: 0 = raw blocks only.  >= 1 uses our native LZ77 + FSE encoder.
/// `content_size`: hint stored in the frame header for the decoder.
pub fn encode_frame(input: &[u8], level: i32, _content_size: Option<u64>) -> Vec<u8> {
    let mut out = Vec::with_capacity(input.len() + 64);

    // -- Frame header --
    out.extend_from_slice(&ZSTD_MAGIC.to_le_bytes());

    let content_checksum = true;
    let single_segment = true; // No window descriptor needed for inputs we handle.
    let fcs_field_size = fcs_field_size_for(input.len() as u64);

    let descriptor: u8 = (fcs_field_size.flag << 6)
        | ((single_segment as u8) << 5)
        | ((content_checksum as u8) << 2);
    out.push(descriptor);

    write_fcs(&mut out, input.len() as u64, fcs_field_size.bytes);

    // -- Blocks --
    if input.is_empty() {
        emit_empty_last_block(&mut out);
    } else if level <= 0 {
        encode_raw_blocks(&mut out, input);
    } else {
        encode_compressed_blocks(&mut out, input, level);
    }

    // -- Content checksum (xxhash64 lower 32 bits) --
    if content_checksum {
        let hash = super::decode::xxhash64_public(input, 0) as u32;
        out.extend_from_slice(&hash.to_le_bytes());
    }

    out
}

/// Worst-case compressed size: raw blocks + frame overhead.
pub fn compress_bound(input_len: usize) -> usize {
    let num_blocks = (input_len + MAX_BLOCK_SIZE - 1) / MAX_BLOCK_SIZE.max(1);
    14 + num_blocks.max(1) * 3 + input_len + 4
}

// =========================================================================
// Raw (uncompressed) blocks — fallback path
// =========================================================================

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

fn emit_empty_last_block(out: &mut Vec<u8>) {
    write_block_header(out, 0, BlockType::Raw, true);
}

#[derive(Copy, Clone)]
#[repr(u8)]
enum BlockType {
    Raw = 0,
    #[allow(dead_code)]
    Rle = 1,
    Compressed = 2,
}

fn write_block_header(out: &mut Vec<u8>, size: usize, ty: BlockType, last: bool) {
    let header = ((size as u32) << 3) | ((ty as u32) << 1) | (last as u32);
    out.push(header as u8);
    out.push((header >> 8) as u8);
    out.push((header >> 16) as u8);
}

// =========================================================================
// Compressed blocks — LZ77 + sequences with predefined FSE tables
// =========================================================================

fn encode_compressed_blocks(out: &mut Vec<u8>, input: &[u8], level: i32) {
    // Match finding is done over the ENTIRE input so block N can back-reference
    // any earlier byte (zstd's window covers the whole input when single_segment).
    // The resulting sequence list is then sliced into ≤128 KB output chunks.
    let parsed = lz77_parse(input, level);

    // Walk sequences, grouping them into output chunks.  A "chunk" here is one
    // zstd compressed block consuming up to MAX_BLOCK_SIZE bytes of output.
    let mut input_pos = 0; // bytes of decompressed output covered so far
    let mut lit_pos = 0; // index into parsed.literals
    let mut seq_idx = 0;
    // Repeat offsets are per-FRAME, not per-block — they must persist across
    // block boundaries to match the decoder's state.
    let mut rep_offsets = [1u32, 4, 8];

    while input_pos < input.len() {
        let chunk_target = input_pos + MAX_BLOCK_SIZE;
        let mut chunk_end = input_pos;
        let block_first_seq = seq_idx;
        let block_first_lit = lit_pos;

        // Pull sequences whose entire footprint fits in this chunk.
        // INVARIANT: every emitted sequence has lit_len + match_len <=
        // MAX_BLOCK_SIZE (enforced by the LZ77 cap on MAX_MATCH).
        while seq_idx < parsed.sequences.len() {
            let s = parsed.sequences[seq_idx];
            let cost = s.lit_len as usize + s.match_len as usize;
            debug_assert!(cost <= MAX_BLOCK_SIZE, "sequence overflows block size");
            if chunk_end + cost > chunk_target {
                break;
            }
            chunk_end += cost;
            lit_pos += s.lit_len as usize;
            seq_idx += 1;
        }

        // Trailing literals (after the last sequence) — only on the final chunk
        // OR when we ran out of sequences.
        let mut trailing_lits = 0usize;
        if seq_idx == parsed.sequences.len() {
            let remaining = input.len() - chunk_end;
            let take = remaining.min(MAX_BLOCK_SIZE - (chunk_end - input_pos));
            trailing_lits = take;
            chunk_end += take;
            // Note: trailing literals come from the parsed.literals tail.
        }

        let last = chunk_end >= input.len();
        let block_seqs = &parsed.sequences[block_first_seq..seq_idx];
        let block_lit_len = (lit_pos - block_first_lit) + trailing_lits;
        let block_lits = &parsed.literals[block_first_lit..block_first_lit + block_lit_len];
        lit_pos += trailing_lits;

        // Try the compressed encoding; fall back to raw if it doesn't shrink.
        // The compressed path mutates `rep_offsets`; if we discard the result we
        // must also restore the rep state so the next block isn't poisoned.
        let raw = &input[input_pos..chunk_end];
        let saved_rep = rep_offsets;
        if let Some(payload) = build_compressed_payload(block_lits, block_seqs, &mut rep_offsets)
        {
            if payload.len() < raw.len() {
                write_block_header(out, payload.len(), BlockType::Compressed, last);
                out.extend_from_slice(&payload);
            } else {
                // Roll back rep_offsets so the discarded compressed encoding
                // doesn't desync the decoder's view.
                rep_offsets = saved_rep;
                write_block_header(out, raw.len(), BlockType::Raw, last);
                out.extend_from_slice(raw);
            }
        } else {
            rep_offsets = saved_rep;
            write_block_header(out, raw.len(), BlockType::Raw, last);
            out.extend_from_slice(raw);
        }

        input_pos = chunk_end;
    }
}

/// Encode literals + sequences as a compressed block payload.
/// Returns None when the slice is empty (no literals AND no sequences).
/// `rep_offsets` carries the per-frame repeat-offset state across blocks
/// (mirrors the decoder's persistent state).
fn build_compressed_payload(
    literals: &[u8],
    sequences: &[Sequence],
    rep_offsets: &mut [u32; 3],
) -> Option<Vec<u8>> {
    if literals.is_empty() && sequences.is_empty() {
        return None;
    }

    let mut payload = Vec::with_capacity(literals.len() + sequences.len() * 4);
    write_raw_literals_section(&mut payload, literals);

    if sequences.is_empty() {
        // Literals-only block: write num_sequences = 0 and stop.
        payload.push(0);
        return Some(payload);
    }

    let seq_section = encode_sequences_section(sequences, rep_offsets)?;
    payload.extend_from_slice(&seq_section);
    Some(payload)
}

// -------------------------------------------------------------------------
// Step 1: LZ77 parsing
// -------------------------------------------------------------------------

#[derive(Debug)]
struct ParsedBlock {
    /// All literal bytes in input order, concatenated.
    literals: Vec<u8>,
    /// Sequences in input order.
    sequences: Vec<Sequence>,
}

/// One LZ77 sequence: lit_run literal bytes followed by a back-reference of
/// `match_len` bytes at distance `offset`.
#[derive(Debug, Clone, Copy)]
struct Sequence {
    lit_len: u32,
    match_len: u32, // real length, >= 3
    offset: u32,    // real distance, >= 1
}

const MIN_MATCH: usize = 3;
/// Cap individual matches to a value strictly less than `MAX_BLOCK_SIZE` so
/// the (lit_len + match_len) total of any single sequence always fits in one
/// block.  Without this cap, a 131_074-byte match plus literals would emit a
/// block whose decompressed size exceeds zstd's 128 KiB block-size limit and
/// the C reference decoder rejects the frame as corrupted (RFC 8878 §3.1.1.2).
const MAX_MATCH: usize = 65_536;
const HASH_BITS: usize = 15;
const HASH_SIZE: usize = 1 << HASH_BITS;
const HASH_MASK: usize = HASH_SIZE - 1;
const NONE: u32 = u32::MAX;
/// Max distance for back references — 1 MiB is plenty for typical inputs and
/// keeps the chain table small.
const MAX_OFFSET: usize = 1 << 20;
/// Window size for the chain table (must be a power of two).
const CHAIN_SIZE: usize = MAX_OFFSET;
const CHAIN_MASK: usize = CHAIN_SIZE - 1;

#[inline]
fn hash4(b: &[u8]) -> usize {
    let v = u32::from_le_bytes([b[0], b[1], b[2], b[3]]);
    (v.wrapping_mul(2654435761) >> (32 - HASH_BITS)) as usize & HASH_MASK
}

/// LZ77 strategy parameters tuned per zstd level.
struct LzConfig {
    /// How many candidates to walk per position.  1 = no chain.
    chain_depth: usize,
    /// Whether to do 1-byte lazy matching.
    lazy: bool,
    /// Insert intermediate hashes inside emitted matches?  Slower but better recall.
    insert_inside_match: bool,
}

fn lz_config(level: i32) -> LzConfig {
    match level {
        ..=1 => LzConfig {
            chain_depth: 1,
            lazy: false,
            insert_inside_match: false,
        },
        2..=3 => LzConfig {
            chain_depth: 2,
            lazy: false,
            insert_inside_match: true,
        },
        4..=6 => LzConfig {
            chain_depth: 4,
            lazy: true,
            insert_inside_match: true,
        },
        _ => LzConfig {
            chain_depth: 8,
            lazy: true,
            insert_inside_match: true,
        },
    }
}

/// LZ77 parse over the entire input with per-level tuning.
fn lz77_parse(input: &[u8], level: i32) -> ParsedBlock {
    let cfg = lz_config(level);
    let len = input.len();
    let mut sequences: Vec<Sequence> = Vec::with_capacity(len / 16);
    let mut literals: Vec<u8> = Vec::with_capacity(len);

    if len < MIN_MATCH + 1 {
        literals.extend_from_slice(input);
        return ParsedBlock { literals, sequences };
    }

    // Hash head + chain (prev pointers).  `prev[pos & CHAIN_MASK]` is the
    // previous position with the same hash.  Allocated once per call.
    let mut head = vec![NONE; HASH_SIZE];
    let mut prev = if cfg.chain_depth > 1 {
        vec![NONE; CHAIN_SIZE]
    } else {
        Vec::new()
    };
    let chain_depth = cfg.chain_depth;

    let mut pos = 0usize;
    let mut lit_run_start = 0usize;

    while pos + 4 <= len {
        // Find the best match at `pos`.
        let (best_off, best_len) = find_best_match(input, pos, &head, &prev, chain_depth);

        // Lazy: peek at pos+1 to see if a longer match starts there.
        let lazy_len = if cfg.lazy && best_len >= MIN_MATCH && pos + 5 <= len {
            // Insert pos so the lazy lookup at pos+1 has it visible.
            insert_hash(&mut head, &mut prev, input, pos);
            let (_, l) = find_best_match(input, pos + 1, &head, &prev, chain_depth);
            l
        } else {
            0
        };

        if best_len >= MIN_MATCH && lazy_len > best_len + 1 {
            // Lazy wins: emit pos as a literal and try the match at pos+1 next.
            pos += 1;
            continue;
        }

        if best_len >= MIN_MATCH {
            // Emit a sequence.
            let lit_len = (pos - lit_run_start) as u32;
            literals.extend_from_slice(&input[lit_run_start..pos]);
            sequences.push(Sequence {
                lit_len,
                match_len: best_len as u32,
                offset: best_off as u32,
            });
            let match_end = pos + best_len;
            // Make sure pos itself is hashed (lazy peek already did this).
            if !cfg.lazy {
                insert_hash(&mut head, &mut prev, input, pos);
            }
            if cfg.insert_inside_match {
                let mut p = pos + 1;
                while p + 4 <= len && p < match_end {
                    insert_hash(&mut head, &mut prev, input, p);
                    p += 1;
                }
            }
            pos = match_end;
            lit_run_start = pos;
        } else {
            insert_hash(&mut head, &mut prev, input, pos);
            pos += 1;
        }
    }

    // Tail literals (no trailing sequence).
    if lit_run_start < len {
        literals.extend_from_slice(&input[lit_run_start..len]);
    }

    ParsedBlock {
        literals,
        sequences,
    }
}

#[inline]
fn insert_hash(head: &mut [u32], prev: &mut [u32], input: &[u8], pos: usize) {
    if pos + 4 > input.len() {
        return;
    }
    let h = hash4(&input[pos..]);
    if !prev.is_empty() {
        prev[pos & CHAIN_MASK] = head[h];
    }
    head[h] = pos as u32;
}

/// Walk the hash chain from `pos` looking for the longest 4-byte (or longer)
/// match within the allowed offset range.  Returns (offset, length).
#[inline]
fn find_best_match(
    input: &[u8],
    pos: usize,
    head: &[u32],
    prev: &[u32],
    max_chain: usize,
) -> (usize, usize) {
    if pos + 4 > input.len() {
        return (0, 0);
    }
    let len = input.len();
    let max_len = MAX_MATCH.min(len - pos);
    let h = hash4(&input[pos..]);
    let mut cand = head[h];
    let mut best_len = 0usize;
    let mut best_off = 0usize;
    let mut chain_left = max_chain;

    while cand != NONE && chain_left > 0 {
        let mp = cand as usize;
        if mp >= pos {
            break;
        }
        let dist = pos - mp;
        if dist > MAX_OFFSET || dist == 0 {
            break;
        }
        if input[mp] == input[pos]
            && input[mp + 1] == input[pos + 1]
            && input[mp + 2] == input[pos + 2]
            && input[mp + 3] == input[pos + 3]
        {
            let mut mlen = 4usize;
            while mlen < max_len && input[mp + mlen] == input[pos + mlen] {
                mlen += 1;
            }
            if mlen > best_len {
                best_len = mlen;
                best_off = dist;
                if mlen >= max_len {
                    break;
                }
            }
        }
        if prev.is_empty() {
            break;
        }
        cand = prev[mp & CHAIN_MASK];
        chain_left -= 1;
    }

    (best_off, best_len)
}

// -------------------------------------------------------------------------
// Step 2: Literals section (raw)
// -------------------------------------------------------------------------

fn write_raw_literals_section(out: &mut Vec<u8>, literals: &[u8]) {
    let regen = literals.len();
    // Choose the smallest size_format that fits.
    if regen < 32 {
        // size_format 00: 1-byte header, 5-bit regen size.
        let header = (regen << 3) as u8 | 0; // lit_type=0, size_format=00
        out.push(header);
    } else if regen < 4096 {
        // size_format 01: 2-byte header, 12-bit regen size.
        // Header byte 0: bits 0-1 = lit_type (0), bits 2-3 = size_format (01),
        // bits 4-7 = low 4 bits of regen.
        let b0 = 0u8 | (1 << 2) | (((regen as u8) & 0x0F) << 4);
        let b1 = (regen >> 4) as u8;
        out.push(b0);
        out.push(b1);
    } else {
        // size_format 11: 3-byte header, 20-bit regen size.
        let b0 = 0u8 | (3 << 2) | (((regen as u8) & 0x0F) << 4);
        let b1 = (regen >> 4) as u8;
        let b2 = (regen >> 12) as u8;
        out.push(b0);
        out.push(b1);
        out.push(b2);
    }
    out.extend_from_slice(literals);
}

// -------------------------------------------------------------------------
// Step 3: Sequence value → (FSE code, extra_bits, extra_value)
// -------------------------------------------------------------------------

/// Map a literals length to its FSE code, extra_bits and extra_value.
fn ll_code(len: u32) -> (u8, u8, u32) {
    // For codes 0..15 the value is direct.
    if len < 16 {
        return (len as u8, 0, 0);
    }
    // Search the table for the code containing this value.
    for code in 16..LITLEN_TABLE.len() {
        let (base, eb) = LITLEN_TABLE[code];
        let span = 1u32 << eb;
        if len >= base && len < base + span {
            return (code as u8, eb, len - base);
        }
    }
    panic!("ll_code: literals length {} too large", len);
}

/// Map a match length (>= 3) to its FSE code.
fn ml_code(len: u32) -> (u8, u8, u32) {
    if len < 35 {
        return ((len - 3) as u8, 0, 0);
    }
    for code in 32..MATCHLEN_TABLE.len() {
        let (base, eb) = MATCHLEN_TABLE[code];
        let span = 1u32 << eb;
        if len >= base && len < base + span {
            return (code as u8, eb, len - base);
        }
    }
    panic!("ml_code: match length {} too large", len);
}

/// Map an offset_value (already including repeat-offset encoding) to its
/// (code, extra_bits, extra_value).  `offset_value` is the raw value the
/// decoder reads (1..=3 for repeats, real_offset+3 otherwise).
fn of_code(offset_value: u32) -> (u8, u8, u32) {
    let code = 31 - offset_value.leading_zeros();
    let extra = offset_value - (1u32 << code);
    (code as u8, code as u8, extra)
}

/// Track the rolling 3 most-recent real offsets and convert (real_offset, lit_len)
/// to the `offset_value` the decoder will read.  Updates `rep` in-place.
///
/// Mirrors `decode::resolve_offset` exactly (RFC 8878 §3.1.1.3.2.1.3).
fn encode_offset(offset: u32, lit_len: u32, rep: &mut [u32; 3]) -> u32 {
    if lit_len > 0 {
        if offset == rep[0] {
            return 1; // no change to rep[]
        }
        if offset == rep[1] {
            rep.swap(0, 1);
            return 2;
        }
        if offset == rep[2] {
            let t = rep[2];
            rep[2] = rep[1];
            rep[1] = rep[0];
            rep[0] = t;
            return 3;
        }
    } else {
        if offset == rep[1] {
            rep.swap(0, 1);
            return 1;
        }
        if offset == rep[2] {
            let t = rep[2];
            rep[2] = rep[1];
            rep[1] = rep[0];
            rep[0] = t;
            return 2;
        }
        if offset + 1 == rep[0] {
            let new = rep[0] - 1;
            rep[2] = rep[1];
            rep[1] = rep[0];
            rep[0] = new;
            return 3;
        }
    }
    // Plain offset.
    rep[2] = rep[1];
    rep[1] = rep[0];
    rep[0] = offset;
    offset + 3
}

// -------------------------------------------------------------------------
// Step 4: Sequence section encoder (predefined FSE tables, mode = 00)
// -------------------------------------------------------------------------

fn encode_sequences_section(seqs: &[Sequence], rep_offsets: &mut [u32; 3]) -> Option<Vec<u8>> {
    let n = seqs.len();
    let mut out = Vec::new();

    // num_sequences encoding (RFC 8878 §3.1.1.3.2.1).
    if n < 128 {
        out.push(n as u8);
    } else if n < 0x7F00 {
        out.push(((n >> 8) | 0x80) as u8);
        out.push((n & 0xFF) as u8);
    } else {
        out.push(0xFF);
        let v = n - 0x7F00;
        out.push((v & 0xFF) as u8);
        out.push((v >> 8) as u8);
    }

    // Symbol_Compression_Modes byte: ll=00, of=00, ml=00, reserved=00.
    out.push(0);

    // Build encoder tables from the predefined decoder tables.
    let ll_dec = fse::predefined_litlen_table();
    let of_dec = fse::predefined_offset_table();
    let ml_dec = fse::predefined_matchlen_table();
    let ll_enc = FseEncoder::from_decoder(&ll_dec, 36);
    let of_enc = FseEncoder::from_decoder(&of_dec, 32);
    let ml_enc = FseEncoder::from_decoder(&ml_dec, 53);

    // Pre-compute (code, extra_bits, extra_value) for every sequence,
    // applying repeat-offset substitution along the way.  `rep_offsets` is
    // owned by the caller so it can persist across blocks.
    let codes: Vec<(SeqCodes, SeqCodes, SeqCodes)> = seqs
        .iter()
        .map(|s| {
            let offset_value = encode_offset(s.offset, s.lit_len, rep_offsets);
            let ll = ll_code(s.lit_len);
            let ml = ml_code(s.match_len);
            let ofc = of_code(offset_value);
            (
                SeqCodes { code: ofc.0, eb: ofc.1, ev: ofc.2 },
                SeqCodes { code: ml.0, eb: ml.1, ev: ml.2 },
                SeqCodes { code: ll.0, eb: ll.1, ev: ll.2 },
            )
        })
        .collect();

    // Encoder writes to a forward bitstream.  In encoder time we process the
    // LAST sequence first; the decoder reads MSB-first from the end which
    // recovers them in input order.
    let mut bw = ForwardBitWriter::with_capacity(seqs.len() * 4);

    // Initial encoder states pinned to the LAST sequence's symbols.
    let last = codes.last().unwrap();
    let mut state_of = of_enc.start_state(last.0.code);
    let mut state_ml = ml_enc.start_state(last.1.code);
    let mut state_ll = ll_enc.start_state(last.2.code);

    // -- Last sequence in encoder time = first encoded.  Only its EXTRA bits go
    //    out (no preceding state-update bits).  Decoder reads of_extra → ml_extra
    //    → ll_extra in this order, so encoder writes them in the REVERSE order:
    //    ll_extra first, then ml_extra, then of_extra.
    write_extra_lmo(&mut bw, &last.2, &last.1, &last.0);

    // -- Walk all preceding sequences in REVERSE encoder time.  For each one,
    //    we (a) emit the state-update bits to "rewind" from sequence (i+1) to
    //    sequence (i), then (b) emit the extra bits for sequence (i).
    //
    //    Decoder bit order between two sequences is ll_update → ml_update →
    //    of_update; encoder writes them in REVERSE: of, ml, ll.
    for i in (0..n - 1).rev() {
        let (of_c, ml_c, ll_c) = &codes[i];
        state_of = of_enc.encode_symbol(state_of, of_c.code, &mut bw);
        state_ml = ml_enc.encode_symbol(state_ml, ml_c.code, &mut bw);
        state_ll = ll_enc.encode_symbol(state_ll, ll_c.code, &mut bw);
        write_extra_lmo(&mut bw, ll_c, ml_c, of_c);
    }

    // -- Initial states.  Decoder reads ll → of → ml; encoder writes ml → of → ll.
    bw.write_bits(state_ml as u64, ml_enc.accuracy_log);
    bw.write_bits(state_of as u64, of_enc.accuracy_log);
    bw.write_bits(state_ll as u64, ll_enc.accuracy_log);

    let bitstream = bw.finalize();
    out.extend_from_slice(&bitstream);

    Some(out)
}

#[derive(Clone, Copy)]
struct SeqCodes {
    code: u8,
    eb: u8,
    ev: u32,
}

#[inline]
fn write_extra_lmo(
    bw: &mut ForwardBitWriter,
    ll: &SeqCodes,
    ml: &SeqCodes,
    ofc: &SeqCodes,
) {
    if ll.eb > 0 {
        bw.write_bits(ll.ev as u64, ll.eb as u32);
    }
    if ml.eb > 0 {
        bw.write_bits(ml.ev as u64, ml.eb as u32);
    }
    if ofc.eb > 0 {
        bw.write_bits(ofc.ev as u64, ofc.eb as u32);
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
    if size <= 255 {
        FcsInfo { flag: 0, bytes: 1 }
    } else if size <= 65535 + 256 {
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
        8 => out.extend_from_slice(&size.to_le_bytes()),
        _ => {}
    }
}

// =========================================================================
// Tests
// =========================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ll_code_basic() {
        assert_eq!(ll_code(0), (0, 0, 0));
        assert_eq!(ll_code(15), (15, 0, 0));
        assert_eq!(ll_code(16), (16, 1, 0));
        assert_eq!(ll_code(17), (16, 1, 1));
        assert_eq!(ll_code(20), (18, 1, 0));
    }

    #[test]
    fn ml_code_basic() {
        assert_eq!(ml_code(3), (0, 0, 0));
        assert_eq!(ml_code(34), (31, 0, 0));
        assert_eq!(ml_code(35), (32, 1, 0));
        assert_eq!(ml_code(36), (32, 1, 1));
    }

    #[test]
    fn of_code_basic() {
        // offset_value 4 → code 2, extra=0
        assert_eq!(of_code(4), (2, 2, 0));
        // offset_value 8 → code 3, extra=0
        assert_eq!(of_code(8), (3, 3, 0));
    }

    #[test]
    fn encode_offset_repeats() {
        let mut rep = [1u32, 4, 8];
        // First-time offset 100 → real, becomes rep[0] = 100, returns 103
        assert_eq!(encode_offset(100, 5, &mut rep), 103);
        assert_eq!(rep, [100, 1, 4]);
        // Offset 100 again with lit_len > 0 → returns 1 (rep[0]), no rep change
        assert_eq!(encode_offset(100, 3, &mut rep), 1);
        assert_eq!(rep, [100, 1, 4]);
        // Offset 1 (= rep[1]) with lit_len > 0 → returns 2, rotates
        assert_eq!(encode_offset(1, 2, &mut rep), 2);
        assert_eq!(rep, [1, 100, 4]);
    }

    #[test]
    fn lz77_parses_repeated_text() {
        let input = b"hello world hello world hello world hello world";
        let parsed = lz77_parse(input, 1);
        assert!(!parsed.sequences.is_empty(), "should find at least one match");
    }

    #[test]
    fn frame_roundtrip_text() {
        let input: Vec<u8> = b"The quick brown fox jumps over the lazy dog. \
                               The quick brown fox jumps over the lazy dog. \
                               The quick brown fox jumps over the lazy dog."
            .to_vec();
        let frame = encode_frame(&input, 1, Some(input.len() as u64));
        let mut decoded = Vec::new();
        let n = super::super::decode::decode_frame(&frame, &mut decoded).unwrap();
        assert_eq!(n, frame.len());
        assert_eq!(decoded, input);
    }

    #[test]
    fn frame_roundtrip_random_falls_back_to_raw() {
        // Pseudo-random data: should not compress, should fall back to raw blocks.
        let mut s: u32 = 0xDEAD_BEEF;
        let input: Vec<u8> = (0..2048)
            .map(|_| {
                s ^= s << 13;
                s ^= s >> 17;
                s ^= s << 5;
                (s >> 16) as u8
            })
            .collect();
        let frame = encode_frame(&input, 1, Some(input.len() as u64));
        let mut decoded = Vec::new();
        super::super::decode::decode_frame(&frame, &mut decoded).unwrap();
        assert_eq!(decoded, input);
    }
}
