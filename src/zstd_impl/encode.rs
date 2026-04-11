//! Zstandard frame encoder (RFC 8878).
//!
//! Two paths:
//!   * Raw blocks for level == 0 (no entropy coding) — used as a fallback when
//!     compression fails to make data smaller.
//!   * LZ77 + sequences-with-predefined-FSE-tables for level >= 1.  Literals are
//!     emitted as raw (lit_type = 0) for now.

use std::cell::RefCell;
use std::sync::LazyLock;

use super::bits::ForwardBitWriter;
use super::fse::{
    self, FseEncoder, FseTable, LITLEN_TABLE, MATCHLEN_TABLE,
    PREDEFINED_LL_WEIGHTS, PREDEFINED_ML_WEIGHTS, PREDEFINED_OF_WEIGHTS,
};
use super::huf::{normalize_to_acc_log, write_fse_table_description};

/// Cache the FSE encoder tables for the predefined LL/OF/ML distributions.
/// They're constant per the spec, so building them once at first use saves
/// three Vec<Vec<...>> rebuilds on every compress() call.
static PREDEFINED_LL_ENC: LazyLock<FseEncoder> =
    LazyLock::new(|| FseEncoder::from_decoder(&fse::predefined_litlen_table(), 36));
static PREDEFINED_OF_ENC: LazyLock<FseEncoder> =
    LazyLock::new(|| FseEncoder::from_decoder(&fse::predefined_offset_table(), 32));
static PREDEFINED_ML_ENC: LazyLock<FseEncoder> =
    LazyLock::new(|| FseEncoder::from_decoder(&fse::predefined_matchlen_table(), 53));

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
    let mut parsed = lz77_parse(input, level);

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
        while seq_idx < parsed.sequences.len() {
            let s = parsed.sequences[seq_idx];
            let cost = s.lit_len as usize + s.match_len as usize;
            if chunk_end + cost > chunk_target {
                break;
            }
            chunk_end += cost;
            lit_pos += s.lit_len as usize;
            seq_idx += 1;
        }

        // If no sequence fit in this chunk AND we have sequences left,
        // the next sequence's `lit_len + match_len > MAX_BLOCK_SIZE`.
        // Split it: emit up to MAX_BLOCK_SIZE bytes as a literals-only
        // block (consuming part of the sequence's lit_len) and shrink
        // the sequence's `lit_len` for the next iteration.
        //
        // This guarantees forward progress.  Without this, the level-1
        // fast parser's skip-ahead can produce sequences with huge
        // lit_len in incompressible regions and the block-emit loop
        // would spin forever emitting empty blocks.
        if seq_idx == block_first_seq && seq_idx < parsed.sequences.len() {
            let s = parsed.sequences[seq_idx];
            let take = (s.lit_len as usize).min(MAX_BLOCK_SIZE);
            debug_assert!(take > 0, "lit_len 0 sequence shouldn't be in the parser output");
            chunk_end += take;
            lit_pos += take;
            // Shrink the sequence's lit_len so the next iteration starts
            // with the residual literals (or directly with the match if
            // we consumed all the literals).
            parsed.sequences[seq_idx].lit_len -= take as u32;
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
    write_literals_section(&mut payload, literals);

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
const HASH_BITS: usize = 17;
const HASH_SIZE: usize = 1 << HASH_BITS;
const HASH_MASK: usize = HASH_SIZE - 1;
/// Level-1 fast path hash table.  Bigger = fewer collisions = more matches
/// found.  128 KiB (2^15 × 4 bytes) is the same order as C zstd's level-1.
const FAST_HASH_BITS: usize = 17;
const FAST_HASH_SIZE: usize = 1 << FAST_HASH_BITS;
const NONE: u32 = u32::MAX;
/// Max distance for back references.
const MAX_OFFSET: usize = 1 << 20;
/// Chain table size (must be a power of two).  Positions that differ by more
/// than CHAIN_SIZE alias in the table, losing older chain links.  128K entries
/// covers most useful back-references and matches C zstd's chainLog=17 at L3-6.
const CHAIN_SIZE: usize = 1 << 17; // 128K entries → 512 KiB
const CHAIN_MASK: usize = CHAIN_SIZE - 1;

#[inline]
fn hash4(b: &[u8]) -> usize {
    let v = u32::from_le_bytes([b[0], b[1], b[2], b[3]]);
    (v.wrapping_mul(2654435761) >> (32 - HASH_BITS)) as usize & HASH_MASK
}

/// Hash for the smaller fast-path table.
#[inline(always)]
fn hash4_fast(v: u32) -> usize {
    (v.wrapping_mul(2654435761) >> (32 - FAST_HASH_BITS)) as usize
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
            chain_depth: 8,
            lazy: false,
            insert_inside_match: true,
        },
        4..=6 => LzConfig {
            chain_depth: 24,
            lazy: true,
            insert_inside_match: true,
        },
        _ => LzConfig {
            chain_depth: 64,
            lazy: true,
            insert_inside_match: true,
        },
    }
}

// Reusable scratch buffers, kept per-thread so the 128 KiB hash table is
// allocated once and reused across compress() calls.
thread_local! {
    static SCRATCH: RefCell<EncoderScratch> = RefCell::new(EncoderScratch::new());
}

/// Per-thread reusable scratch state for the encoder.
///
/// The fast path uses `(fast_head, fast_gen)`: each entry is the position
/// of the most recent occurrence and the generation it was inserted in.
/// To "reset" between calls we just bump `current_gen`, avoiding a 32 KiB
/// memset on every compress() call.  When the generation counter wraps we
/// do a single full clear.
struct EncoderScratch {
    /// Generic-path head table (full size, walked alongside `prev`).
    head: Vec<u32>,
    /// Generic-path back-pointer chain (one slot per `pos & CHAIN_MASK`).
    prev: Vec<u32>,
    /// Fast-path head table (smaller, lazily reset via generation counter).
    fast_head: Vec<u32>,
    fast_gen: Vec<u32>,
    /// Generation token for the fast path.  Bumped per call.
    fast_current_gen: u32,
}

impl EncoderScratch {
    fn new() -> Self {
        Self {
            head: Vec::new(),
            prev: Vec::new(),
            fast_head: Vec::new(),
            fast_gen: Vec::new(),
            fast_current_gen: 0,
        }
    }

    /// Resize and reset the generic-path head/prev tables.
    fn reset_generic(&mut self, with_chain: bool) {
        if self.head.len() != HASH_SIZE {
            self.head.clear();
            self.head.resize(HASH_SIZE, NONE);
        } else {
            for slot in self.head.iter_mut() {
                *slot = NONE;
            }
        }
        if with_chain {
            if self.prev.len() != CHAIN_SIZE {
                self.prev.clear();
                self.prev.resize(CHAIN_SIZE, NONE);
            } else {
                for slot in self.prev.iter_mut() {
                    *slot = NONE;
                }
            }
        } else {
            self.prev.clear();
        }
    }

    /// "Reset" the fast-path table by bumping the generation counter.
    /// Allocates the table on first use.  Wraps with a full clear.
    fn reset_fast(&mut self) {
        if self.fast_head.is_empty() {
            self.fast_head.resize(FAST_HASH_SIZE, 0);
            self.fast_gen.resize(FAST_HASH_SIZE, 0);
            self.fast_current_gen = 1;
            return;
        }
        if self.fast_current_gen == u32::MAX {
            for slot in self.fast_gen.iter_mut() {
                *slot = 0;
            }
            self.fast_current_gen = 1;
        } else {
            self.fast_current_gen += 1;
        }
    }
}

/// Update the parser's repeat-offset state after emitting a sequence.
/// Mirrors the logic in `encode_offset` but only performs the state
/// mutation — the return value is the zstd-encoded offset (not used by
/// the parser, but useful for assertions).
#[inline]
fn update_rep(rep: &mut [u32; 3], offset: u32, lit_len: u32) {
    // Delegate to encode_offset for the state mutation; discard the
    // encoded value — we only want the side-effect on `rep`.
    let _ = encode_offset(offset, lit_len, rep);
}

/// Try to match at `pos` using a repeat offset.  Returns (match_len, rep_index)
/// of the best repeat match, or (0, 0) if none reaches MIN_MATCH.
#[inline]
fn try_rep_match(input: &[u8], pos: usize, rep: &[u32; 3]) -> (usize, usize) {
    let len = input.len();
    let max = MAX_MATCH.min(len - pos);
    let mut best_len = 0usize;
    let mut best_idx = 0usize;
    for (ri, &r) in rep.iter().enumerate() {
        let d = r as usize;
        if d == 0 || d > pos {
            continue;
        }
        let mp = pos - d;
        // Quick 4-byte prefix test.
        if pos + 4 <= len && mp + 4 <= len && u32_at(input, mp) == u32_at(input, pos) {
            let mlen = 4 + count_common_bytes(input, mp + 4, pos + 4, max - 4);
            if mlen > best_len {
                best_len = mlen;
                best_idx = ri;
            }
        }
    }
    (best_len, best_idx)
}

/// LZ77 parse over the entire input with per-level tuning.
fn lz77_parse(input: &[u8], level: i32) -> ParsedBlock {
    if level <= 1 {
        return SCRATCH.with(|s| lz77_parse_fast(input, &mut s.borrow_mut()));
    }
    // NOTE: lz77_parse_optimal exists but its price model is too rough to
    // beat the tuned general parser — it needs per-block FSE cost estimates
    // and rep-offset tracking within the DP window.  Disabled until those
    // are implemented.
    SCRATCH.with(|s| lz77_parse_general(input, level, &mut s.borrow_mut()))
}

fn lz77_parse_general(input: &[u8], level: i32, scratch: &mut EncoderScratch) -> ParsedBlock {
    let cfg = lz_config(level);
    let len = input.len();
    let mut sequences: Vec<Sequence> = Vec::with_capacity(len / 16);
    let mut literals: Vec<u8> = Vec::with_capacity(len);

    if len < MIN_MATCH + 1 {
        literals.extend_from_slice(input);
        return ParsedBlock { literals, sequences };
    }

    scratch.reset_generic(cfg.chain_depth > 1);
    let head = &mut scratch.head;
    let prev = &mut scratch.prev;
    let chain_depth = cfg.chain_depth;
    let mut rep = [1u32, 4, 8];

    let mut pos = 0usize;
    let mut lit_run_start = 0usize;

    while pos + 4 <= len {
        // Check repeat offsets first — they're very cheap to encode.
        let (rep_len, rep_idx) = try_rep_match(input, pos, &rep);
        if rep_len >= MIN_MATCH {
            let lit_len = (pos - lit_run_start) as u32;
            let d = rep[rep_idx];
            literals.extend_from_slice(&input[lit_run_start..pos]);
            sequences.push(Sequence { lit_len, match_len: rep_len as u32, offset: d });
            insert_hash(head, prev, input, pos);
            update_rep(&mut rep, d, lit_len);
            pos += rep_len;
            lit_run_start = pos;
            continue;
        }

        // Find the best match at `pos`.
        let (best_off, best_len) = find_best_match(input, pos, head, prev, chain_depth);

        // Lazy: peek at pos+1 to see if a longer match starts there.
        let lazy_len = if cfg.lazy && best_len >= MIN_MATCH && pos + 5 <= len {
            // Insert pos so the lazy lookup at pos+1 has it visible.
            insert_hash(head, prev, input, pos);
            let (_, l) = find_best_match(input, pos + 1, head, prev, chain_depth);
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
            update_rep(&mut rep, best_off as u32, lit_len);
            let match_end = pos + best_len;
            // Make sure pos itself is hashed (lazy peek already did this).
            if !cfg.lazy {
                insert_hash(head, prev, input, pos);
            }
            // Insert hashes inside the match so future positions can
            // chain through them.  Cap insertions to avoid O(N) cost on
            // very long matches (64 KiB+).
            if cfg.insert_inside_match {
                let insert_limit = if chain_depth >= 24 { 512 } else { 64 };
                if best_len < insert_limit {
                    let mut p = pos + 1;
                    while p + 4 <= len && p < match_end {
                        insert_hash(head, prev, input, p);
                        p += 1;
                    }
                }
            }
            pos = match_end;
            lit_run_start = pos;
        } else {
            insert_hash(head, prev, input, pos);
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

// =========================================================================
// Optimal parser — forward DP with limited-window lookahead (L7+)
// Disabled: the price model needs per-block FSE costs and rep-offset
// tracking within windows to beat the tuned greedy/lazy parser.
// =========================================================================

#[allow(dead_code)]
/// Window size for the forward DP.  Larger = better decisions but slower.
const OPT_WINDOW: usize = 64;
#[allow(dead_code)]
/// Max chain depth for match finding INSIDE the DP.  Kept lower than the
/// outer encoder's chain_depth to bound DP cost: each window position does
/// one chain walk of up to this depth.
const OPT_CHAIN_DEPTH: usize = 16;

/// DP table entry.  Tracks the best known way to reach this position.
#[derive(Clone, Copy)]
#[allow(dead_code)]
struct OptNode {
    /// Total cost (in approximate bits) to encode from the window start.
    cost: u32,
    /// How we reached this position: 0 = literal, >0 = match of this length.
    mlen: u32,
    /// Offset for the match (valid only when mlen > 0).
    off: u32,
    /// Number of literal bytes accumulated before the match at this node.
    lits: u32,
}

impl Default for OptNode {
    fn default() -> Self {
        Self { cost: u32::MAX, mlen: 0, off: 0, lits: 0 }
    }
}

#[allow(dead_code)]
/// Estimate the cost (in bits) of a sequence header: LL code + OF code + ML code.
/// This does NOT include the literal bytes themselves (those are accounted for
/// separately as LIT_COST per byte).
#[inline]
fn seq_price(lit_len: u32, offset_value: u32, match_len: u32) -> u32 {
    // LL: FSE state transition (~5 bits) + extra bits from the code table.
    let (_, ll_eb, _) = ll_code(lit_len);
    // OF: FSE state transition (~5 bits) + floor(log2(offset_value)) extra bits.
    let of_bits = 32 - offset_value.max(1).leading_zeros();
    // ML: FSE state transition (~5 bits) + extra bits.
    let (_, ml_eb, _) = ml_code(match_len);
    // Total: 3 FSE updates (~15 bits) + extra bits.
    15 + ll_eb as u32 + of_bits + ml_eb as u32
}

#[allow(dead_code)]
/// Approximate bits per literal byte.
const LIT_COST: u32 = 8;

#[allow(dead_code)]
/// Try a match of length `mlen` at offset `off` from DP position `i`.
/// Updates the DP table for the full match length and a few shorter lengths.
#[inline]
fn opt_try_match(
    opt: &mut [OptNode], i: usize, wlen: usize, off: u32, mlen: usize,
    cur_cost: u32, cur_lits: u32, ov: u32,
) {
    // Try the full match length and shorter lengths down to MIN_MATCH.
    // For very long matches, only try a few lengths near the top and bottom
    // to keep the loop bounded.
    let min_try = if mlen > 16 { mlen - 8 } else { MIN_MATCH };
    for ml in (min_try..=mlen).rev() {
        let end = i + ml;
        if end > wlen {
            continue;
        }
        let cost = cur_cost + seq_price(cur_lits, ov, ml as u32);
        if cost < opt[end].cost {
            opt[end] = OptNode {
                cost,
                mlen: ml as u32,
                off,
                lits: 0,
            };
        }
    }
    // Also try MIN_MATCH if we didn't reach it above.
    if min_try > MIN_MATCH {
        let end = i + MIN_MATCH;
        if end <= wlen {
            let cost = cur_cost + seq_price(cur_lits, ov, MIN_MATCH as u32);
            if cost < opt[end].cost {
                opt[end] = OptNode {
                    cost,
                    mlen: MIN_MATCH as u32,
                    off,
                    lits: 0,
                };
            }
        }
    }
}

/// Forward-DP optimal parser.  Processes the input in overlapping windows of
/// `OPT_WINDOW` positions, selecting minimum-cost parse decisions.
#[allow(dead_code)]
fn lz77_parse_optimal(input: &[u8], level: i32, scratch: &mut EncoderScratch) -> ParsedBlock {
    let len = input.len();
    let cfg = lz_config(level);
    let mut sequences: Vec<Sequence> = Vec::with_capacity(len / 16);
    let mut literals: Vec<u8> = Vec::with_capacity(len);

    if len < MIN_MATCH + 4 {
        literals.extend_from_slice(input);
        return ParsedBlock { literals, sequences };
    }

    scratch.reset_generic(cfg.chain_depth > 1);
    let head = &mut scratch.head;
    let prev = &mut scratch.prev;
    let chain_depth = cfg.chain_depth;
    let mut rep = [1u32, 4, 8];

    let mut pos = 0usize;
    let mut lit_run_start = 0usize;
    let mut opt_buf = vec![OptNode::default(); OPT_WINDOW + 1];

    while pos + 4 <= len {
        // Quick check: is there any match here at all?
        // Search BEFORE hashing pos — if we hash first, pos itself
        // becomes the chain head and find_best_match breaks immediately.
        let (_, best_len) = find_best_match(input, pos, head, prev, chain_depth);
        let (rep_len, _) = try_rep_match(input, pos, &rep);
        if best_len < MIN_MATCH && rep_len < MIN_MATCH {
            insert_hash(head, prev, input, pos);
            pos += 1;
            continue;
        }

        // There's at least one match — run the DP window from here.
        let window_end = (pos + OPT_WINDOW).min(len);
        let wlen = window_end - pos;

        // Reset the DP table.
        for i in 0..=wlen {
            opt_buf[i] = OptNode::default();
        }
        opt_buf[0] = OptNode { cost: 0, mlen: 0, off: 0, lits: 0 };

        for i in 0..wlen {
            if opt_buf[i].cost == u32::MAX {
                // Still hash this position so later positions can find it.
                let ip = pos + i;
                if ip + 4 <= len {
                    insert_hash(head, prev, input, ip);
                }
                continue;
            }
            let ip = pos + i;

            let cur_cost = opt_buf[i].cost;
            let cur_lits = opt_buf[i].lits;

            // Option 1: emit a literal at position ip.
            if i + 1 <= wlen {
                let new_cost = cur_cost + LIT_COST;
                if new_cost < opt_buf[i + 1].cost {
                    opt_buf[i + 1] = OptNode {
                        cost: new_cost,
                        mlen: 0,
                        off: 0,
                        lits: cur_lits + 1,
                    };
                }
            }

            if ip + 4 > len {
                continue; // can't hash or match with < 4 bytes
            }

            // Option 2: repeat-offset matches at position ip.
            for ri in 0..3 {
                let d = rep[ri] as usize;
                if d == 0 || d > ip {
                    continue;
                }
                let mp = ip - d;
                if mp + 4 <= len && u32_at(input, mp) == u32_at(input, ip) {
                    let max = MAX_MATCH.min(len - ip);
                    let mlen = 4 + count_common_bytes(input, mp + 4, ip + 4, max - 4);
                    let ov = encode_offset_value(d as u32, cur_lits, &rep);
                    // Try full length and a few shorter.
                    opt_try_match(&mut opt_buf, i, wlen, d as u32, mlen,
                                  cur_cost, cur_lits, ov);
                }
            }

            // Option 3: walk the hash chain to find MULTIPLE matches at
            // different offsets.  Each chain entry that extends the best
            // known length is a new candidate for the DP.
            if ip + 4 <= len {
                let h = hash4(&input[ip..]);
                let mut cand = head[h];
                let mut chain_left = OPT_CHAIN_DEPTH;
                let mut prev_best = 0usize;

                while cand != NONE && chain_left > 0 {
                    let mp = cand as usize;
                    if mp >= ip {
                        break;
                    }
                    let dist = ip - mp;
                    if dist > MAX_OFFSET || dist == 0 {
                        break;
                    }
                    if u32_at(input, mp) == u32_at(input, ip) {
                        let max_ml = MAX_MATCH.min(len - ip);
                        let mlen = 4 + count_common_bytes(
                            input, mp + 4, ip + 4, max_ml - 4,
                        );
                        // Only consider if this match extends beyond previous
                        // candidates (otherwise it's strictly dominated).
                        if mlen > prev_best {
                            let ov = encode_offset_value(
                                dist as u32, cur_lits, &rep,
                            );
                            opt_try_match(
                                &mut opt_buf, i, wlen, dist as u32, mlen,
                                cur_cost, cur_lits, ov,
                            );
                            prev_best = mlen;
                            if mlen >= max_ml {
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
            }

            // Hash ip AFTER all match finding is done, so the next
            // iteration (at ip+1) can find ip in the chain.
            insert_hash(head, prev, input, ip);
        }

        // Pick the end of the window: the furthest position that the DP
        // reached.  This always processes the full window (or as far as
        // we can get), avoiding the problem of committing to tiny chunks.
        let mut best_end = wlen;
        while best_end > 0 && opt_buf[best_end].cost == u32::MAX {
            best_end -= 1;
        }
        if best_end == 0 {
            // No match reachable (shouldn't happen since we checked above).
            pos += 1;
            continue;
        }

        // Backtrack through the DP to reconstruct the parse decisions.
        let mut decisions: Vec<(u32, u32)> = Vec::new();
        let mut i = best_end;
        while i > 0 {
            let node = opt_buf[i];
            if node.mlen > 0 {
                decisions.push((node.mlen, node.off));
                i -= node.mlen as usize;
            } else {
                decisions.push((0, 0));
                i -= 1;
            }
        }
        decisions.reverse();

        // Emit the decisions as sequences + literals.
        let mut dpos = pos;
        for &(mlen, off) in &decisions {
            if mlen == 0 {
                dpos += 1;
            } else {
                let lit_len = (dpos - lit_run_start) as u32;
                literals.extend_from_slice(&input[lit_run_start..dpos]);
                sequences.push(Sequence {
                    lit_len,
                    match_len: mlen,
                    offset: off,
                });
                update_rep(&mut rep, off, lit_len);
                dpos += mlen as usize;
                lit_run_start = dpos;
            }
        }
        pos = dpos;
    }

    // Tail literals.
    if lit_run_start < len {
        literals.extend_from_slice(&input[lit_run_start..len]);
    }

    ParsedBlock { literals, sequences }
}

/// Compute the offset_value the decoder would see, WITHOUT mutating rep state.
/// Used by the price estimator in the optimal parser.
#[allow(dead_code)]
#[inline]
fn encode_offset_value(offset: u32, lit_len: u32, rep: &[u32; 3]) -> u32 {
    if lit_len > 0 {
        if offset == rep[0] { return 1; }
        if offset == rep[1] { return 2; }
        if offset == rep[2] { return 3; }
    } else {
        if offset == rep[1] { return 1; }
        if offset == rep[2] { return 2; }
        if offset + 1 == rep[0] { return 3; }
    }
    offset + 3
}

#[inline(always)]
fn insert_hash(head: &mut [u32], prev: &mut [u32], input: &[u8], pos: usize) {
    if pos + 4 > input.len() {
        return;
    }
    // Inline hash4 to avoid a slice re-bound and use direct unchecked access.
    let v = unsafe {
        std::ptr::read_unaligned(input.as_ptr().add(pos) as *const u32)
    };
    let h = (v.wrapping_mul(2654435761) >> (32 - HASH_BITS)) as usize & HASH_MASK;
    unsafe {
        if !prev.is_empty() {
            *prev.get_unchecked_mut(pos & CHAIN_MASK) = *head.get_unchecked(h);
        }
        *head.get_unchecked_mut(h) = pos as u32;
    }
}

/// Specialized fast path for level 1.  No chain, no lazy match, aggressive
/// skip-ahead in literal runs.  Mirrors the structure of zstd's
/// `ZSTD_compressBlock_fast`.
///
/// Uses a small thread-local hash table with a generation counter so a
/// "reset" between calls is O(1) — just bumping `current_gen` — instead of
/// memset-ing the table.  Each entry is a (position, generation) pair.
///
/// Skip-ahead: the literal-run length scales the step size, so uncompressible
/// regions are O(N/step) instead of O(N).
fn lz77_parse_fast(input: &[u8], scratch: &mut EncoderScratch) -> ParsedBlock {
    let len = input.len();
    let lit_cap = (len / 4).max(64);
    let mut sequences: Vec<Sequence> = Vec::with_capacity((len / 32).max(8));
    let mut literals: Vec<u8> = Vec::with_capacity(lit_cap);

    if len < MIN_MATCH + 4 {
        literals.extend_from_slice(input);
        return ParsedBlock { literals, sequences };
    }

    scratch.reset_fast();
    let gen_now = scratch.fast_current_gen;
    let head = scratch.fast_head.as_mut_slice();
    let gen = scratch.fast_gen.as_mut_slice();

    // Parser-local repeat-offset state (mirrors the decoder's initial state).
    let mut rep = [1u32, 4, 8];

    let mut lit_run_start = 0usize;
    let stop = len - 4;

    // Seed position 0.
    let v0 = u32_at(input, 0);
    let h0 = hash4_fast(v0);
    unsafe {
        *head.get_unchecked_mut(h0) = 0;
        *gen.get_unchecked_mut(h0) = gen_now;
    }

    /// Emit a repeat-offset continuation chain: keep matching at the current
    /// position using rep offsets and emit lit_len=0 sequences.
    ///
    /// IMPORTANT: for lit_len=0 sequences, the zstd offset encoding is
    /// shifted — code 1 means rep[1] (not rep[0]).  So we probe rep[1]
    /// here: a match at rep[1] encodes as code 1 (cheapest), which also
    /// swaps rep[0]/rep[1], putting the new match's offset into rep[0]
    /// for the next iteration.
    #[inline]
    fn emit_rep_continuations(
        input: &[u8], pos: &mut usize, stop: usize,
        rep: &mut [u32; 3], sequences: &mut Vec<Sequence>,
        lit_run_start: &mut usize,
    ) {
        while *pos <= stop {
            // For lit_len == 0: code 1 → rep[1], code 2 → rep[2].
            // Check rep[1] first (cheapest encoding).
            let d = rep[1] as usize;
            if d > 0 && d <= *pos && *pos + 4 <= input.len() {
                let mp = *pos - d;
                if u32_at(input, mp) == u32_at(input, *pos) {
                    let max = MAX_MATCH.min(input.len() - *pos);
                    let cont_len = 4 + count_common_bytes(input, mp + 4, *pos + 4, max - 4);
                    if cont_len >= MIN_MATCH {
                        let offset = d as u32;
                        sequences.push(Sequence { lit_len: 0, match_len: cont_len as u32, offset });
                        update_rep(rep, offset, 0);
                        *pos += cont_len;
                        *lit_run_start = *pos;
                        continue;
                    }
                }
            }
            break;
        }
    }

    let mut pos = 1usize;

    while pos <= stop {
        // --- 1. Check repeat offsets FIRST (they encode very cheaply) ---
        let (rep_len, rep_idx) = try_rep_match(input, pos, &rep);
        if rep_len >= MIN_MATCH {
            let lit_len = (pos - lit_run_start) as u32;
            let d = rep[rep_idx];
            literals.extend_from_slice(&input[lit_run_start..pos]);
            sequences.push(Sequence { lit_len, match_len: rep_len as u32, offset: d });
            update_rep(&mut rep, d, lit_len);
            pos += rep_len;
            lit_run_start = pos;
            emit_rep_continuations(input, &mut pos, stop, &mut rep, &mut sequences, &mut lit_run_start);
            continue;
        }

        // --- 2. Hash lookup ---
        let v = u32_at(input, pos);
        let h = hash4_fast(v);

        let (cand_pos, cand_gen) = unsafe {
            (*head.get_unchecked(h), *gen.get_unchecked(h))
        };
        unsafe {
            *head.get_unchecked_mut(h) = pos as u32;
            *gen.get_unchecked_mut(h) = gen_now;
        }

        if cand_gen == gen_now {
            let mp = cand_pos as usize;
            let dist = pos - mp;
            if dist <= MAX_OFFSET && dist > 0 {
                let a = u32_at(input, mp);
                if a == v {
                    let max = MAX_MATCH.min(len - pos);
                    let mut mlen = 4 + count_common_bytes(input, mp + 4, pos + 4, max - 4);

                    let mut start = pos;
                    let mut back_mp = mp;
                    while start > lit_run_start
                        && back_mp > 0
                        && mlen < MAX_MATCH
                        && unsafe { *input.get_unchecked(back_mp - 1) }
                            == unsafe { *input.get_unchecked(start - 1) }
                    {
                        start -= 1;
                        back_mp -= 1;
                        mlen += 1;
                    }
                    debug_assert!(mlen <= MAX_MATCH);
                    debug_assert!(start + mlen > pos);

                    let lit_len = (start - lit_run_start) as u32;
                    literals.extend_from_slice(&input[lit_run_start..start]);
                    let d = (start - back_mp) as u32;
                    sequences.push(Sequence { lit_len, match_len: mlen as u32, offset: d });
                    update_rep(&mut rep, d, lit_len);
                    pos = start + mlen;
                    lit_run_start = pos;
                    emit_rep_continuations(input, &mut pos, stop, &mut rep, &mut sequences, &mut lit_run_start);
                    continue;
                }
            }
        }

        // Skip-ahead with rep0 check at every visited position.  The hash
        // lookup is skipped (that's the "fast" part) but the rep0 probe is
        // just one 4-byte compare — nearly free.
        let miss_count = pos - lit_run_start;
        let step = 1 + (miss_count >> 6);
        pos += step;

        // Check rep0 at the new (skipped-to) position.
        if pos <= stop {
            let (rep_len, rep_idx) = try_rep_match(input, pos, &rep);
            if rep_len >= MIN_MATCH {
                let lit_len = (pos - lit_run_start) as u32;
                let d = rep[rep_idx];
                literals.extend_from_slice(&input[lit_run_start..pos]);
                sequences.push(Sequence { lit_len, match_len: rep_len as u32, offset: d });
                update_rep(&mut rep, d, lit_len);
                pos += rep_len;
                lit_run_start = pos;
                emit_rep_continuations(input, &mut pos, stop, &mut rep, &mut sequences, &mut lit_run_start);
            }
        }
    }

    if lit_run_start < len {
        literals.extend_from_slice(&input[lit_run_start..len]);
    }

    ParsedBlock { literals, sequences }
}

#[inline(always)]
fn u32_at(input: &[u8], idx: usize) -> u32 {
    debug_assert!(idx + 4 <= input.len());
    unsafe {
        let p = input.as_ptr().add(idx) as *const u32;
        std::ptr::read_unaligned(p)
    }
}

#[inline(always)]
fn u64_at(input: &[u8], idx: usize) -> u64 {
    debug_assert!(idx + 8 <= input.len());
    unsafe {
        let p = input.as_ptr().add(idx) as *const u64;
        std::ptr::read_unaligned(p)
    }
}

/// Count how many bytes match starting from `(a, b)`, up to `max`.
/// Loads 8 bytes at a time and uses XOR + trailing_zeros to find the first
/// differing byte (little-endian assumption — true on all targets cramjam ships
/// to: x86_64, aarch64, wasm32).
#[inline(always)]
fn count_common_bytes(input: &[u8], a: usize, b: usize, max: usize) -> usize {
    use core::simd::cmp::SimdPartialEq;
    use core::simd::u8x32;

    let mut n = 0usize;

    // First 8 bytes via u64 — handles the common short-match case (most
    // hash hits extend by only 0-7 bytes) without SIMD setup overhead.
    if n + 8 <= max {
        let x = u64_at(input, a + n) ^ u64_at(input, b + n);
        if x != 0 {
            return n + (x.trailing_zeros() as usize / 8);
        }
        n += 8;
    }

    // For longer matches, escalate to 32-byte SIMD chunks.
    while n + 32 <= max {
        let va = u8x32::from_slice(unsafe {
            std::slice::from_raw_parts(input.as_ptr().add(a + n), 32)
        });
        let vb = u8x32::from_slice(unsafe {
            std::slice::from_raw_parts(input.as_ptr().add(b + n), 32)
        });
        let bm = va.simd_ne(vb).to_bitmask();
        if bm != 0 {
            return n + bm.trailing_zeros() as usize;
        }
        n += 32;
    }

    // 8-byte u64 tail.
    while n + 8 <= max {
        let x = u64_at(input, a + n) ^ u64_at(input, b + n);
        if x == 0 {
            n += 8;
        } else {
            return n + (x.trailing_zeros() as usize / 8);
        }
    }
    while n < max && unsafe { *input.get_unchecked(a + n) == *input.get_unchecked(b + n) } {
        n += 1;
    }
    n
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
        // u32 prefix compare via single load.
        if u32_at(input, mp) == u32_at(input, pos) {
            let mlen = 4 + count_common_bytes(input, mp + 4, pos + 4, max_len - 4);
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
// Step 2: Literals section
// -------------------------------------------------------------------------

/// Choose between raw and Huffman literals based on size and try to write
/// the smaller of the two.  Always falls back to raw if Huffman fails or
/// would be larger.
fn write_literals_section(out: &mut Vec<u8>, literals: &[u8]) {
    // Huffman has overhead (~256 bytes for the table) so it's only worth
    // it for moderately sized literal pools.  Below ~1 KiB the raw header
    // wins.
    if literals.len() >= 1024 {
        if let Some(start) = try_write_huffman_literals(out, literals) {
            let huff_size = out.len() - start;
            let raw_size = raw_literals_size(literals.len()) + literals.len();
            if huff_size < raw_size {
                return; // Huffman wins, leave it in `out`.
            }
            out.truncate(start);
        }
    }
    write_raw_literals_section(out, literals);
}

fn raw_literals_size(regen: usize) -> usize {
    if regen < 32 {
        1
    } else if regen < 4096 {
        2
    } else {
        3
    }
}

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

/// Try to encode a Huffman literals section into `out`.  Returns the starting
/// index of the section on success so the caller can compare sizes / roll back.
///
/// Picks the smallest size_format that fits.  Uses 1-stream for tiny literal
/// pools and 4-stream for everything else (zstd's standard layout).  Direct
/// (4-bit) weight serialization is used; FSE-compressed weights would shave
/// a few bytes off the table description but cost extra bitstream encoding.
fn try_write_huffman_literals(out: &mut Vec<u8>, literals: &[u8]) -> Option<usize> {
    let start = out.len();

    // Build the Huffman encoder table from the literal frequencies.
    let enc = super::huf::HufEncoder::from_data(literals)?;
    let weights = enc.weights();
    if weights.len() < 2 {
        return None;
    }
    // Drop the implicit last weight — the decoder infers it from leftover.
    let weights_to_emit = &weights[..weights.len() - 1];

    // Choose between direct (<=128 weights) and FSE-compressed weight
    // encoding.  Direct uses the (header_byte = 127 + N) form with packed
    // 4-bit weights; FSE uses (header_byte = compressed_size < 128).
    let (huff_desc, huff_desc_len) = if weights_to_emit.len() <= 128 {
        let n = weights_to_emit.len();
        (HuffDesc::Direct, 1 + (n + 1) / 2)
    } else if let Some(fse_bytes) = super::huf::encode_weights_fse(weights_to_emit) {
        let len = 1 + fse_bytes.len();
        (HuffDesc::FseCompressed(fse_bytes), len)
    } else {
        return None;
    };
    let regen = literals.len();

    // Decide stream layout: 1-stream is only allowed when regen < 1024 AND the
    // compressed size also fits in 10 bits.  Try the 1-stream path first;
    // otherwise produce 4 independent streams + a 6-byte jump table.
    let mut bw = super::bits::ForwardBitWriter::with_capacity(literals.len());
    enc.encode_stream(&mut bw, literals);
    let single_stream = bw.finalize();

    let one_stream_total = huff_desc_len + single_stream.len();

    if regen < 1024 && one_stream_total < 1024 {
        // size_format 00: 1-stream, 10-bit regen, 10-bit compressed (3 bytes header).
        let combined: u32 = 2u32
            | (0u32 << 2)
            | ((regen as u32) << 4)
            | ((one_stream_total as u32) << 14);
        out.push(combined as u8);
        out.push((combined >> 8) as u8);
        out.push((combined >> 16) as u8);
        write_huffman_desc(out, &huff_desc, weights_to_emit);
        out.extend_from_slice(&single_stream);
        return Some(start);
    }

    // 4-stream layout — split literals into 4 quarters.  Each stream gets
    // its OWN backward bitstream (sentinel + padding).
    if literals.len() < 4 {
        return None;
    }
    let split = (literals.len() + 3) / 4;
    let chunks: [&[u8]; 4] = [
        &literals[..split],
        &literals[split..(split * 2).min(literals.len())],
        &literals[(split * 2).min(literals.len())..(split * 3).min(literals.len())],
        &literals[(split * 3).min(literals.len())..],
    ];

    let mut streams: [Vec<u8>; 4] = [Vec::new(), Vec::new(), Vec::new(), Vec::new()];
    for i in 0..4 {
        let mut bw = super::bits::ForwardBitWriter::with_capacity(chunks[i].len());
        enc.encode_stream(&mut bw, chunks[i]);
        streams[i] = bw.finalize();
    }
    let s1 = streams[0].len();
    let s2 = streams[1].len();
    let s3 = streams[2].len();
    let s4 = streams[3].len();
    if s1 > u16::MAX as usize || s2 > u16::MAX as usize || s3 > u16::MAX as usize {
        return None;
    }
    // 6 bytes of jump table + the 4 streams.
    let four_stream_total = huff_desc_len + 6 + s1 + s2 + s3 + s4;

    // Pick the smallest size_format that fits.
    let (size_format, header_bytes): (u8, usize) =
        if regen < 1024 && four_stream_total < 1024 {
            (1, 3)
        } else if regen < 16_384 && four_stream_total < 16_384 {
            (2, 4)
        } else if regen < 262_144 && four_stream_total < 262_144 {
            (3, 5)
        } else {
            return None;
        };

    let lit_type: u8 = 2;
    match header_bytes {
        3 => {
            let combined: u32 =
                lit_type as u32 | ((size_format as u32) << 2) | ((regen as u32) << 4) | ((four_stream_total as u32) << 14);
            out.push(combined as u8);
            out.push((combined >> 8) as u8);
            out.push((combined >> 16) as u8);
        }
        4 => {
            let combined: u32 =
                lit_type as u32 | ((size_format as u32) << 2) | ((regen as u32) << 4) | ((four_stream_total as u32) << 18);
            out.push(combined as u8);
            out.push((combined >> 8) as u8);
            out.push((combined >> 16) as u8);
            out.push((combined >> 24) as u8);
        }
        5 => {
            // Combined value spans more than 32 bits — split.
            let lo: u32 =
                lit_type as u32 | ((size_format as u32) << 2) | ((regen as u32 & 0xFFFFFFF) << 4);
            // Top 4 bits of regen go into byte 4 low nibble; compressed size starts there.
            let hi_regen_bits = regen >> 28;
            // Byte layout per RFC 8878 §3.1.1.3.1.2:
            //   byte0: bits 0-3 = lit_type|size_format, bits 4-7 = regen[0..4]
            //   byte1: regen[4..12]
            //   byte2: regen[12..18] | compressed[0..2]
            //   byte3: compressed[2..10]
            //   byte4: compressed[10..18]
            let b0 = (lit_type | (size_format << 2) | (((regen as u8) & 0x0F) << 4)) as u8;
            let b1 = ((regen >> 4) & 0xFF) as u8;
            let b2 = (((regen >> 12) & 0x3F) | ((four_stream_total & 0x3) << 6)) as u8;
            let b3 = ((four_stream_total >> 2) & 0xFF) as u8;
            let b4 = ((four_stream_total >> 10) & 0xFF) as u8;
            out.push(b0);
            out.push(b1);
            out.push(b2);
            out.push(b3);
            out.push(b4);
            let _ = lo;
            let _ = hi_regen_bits;
        }
        _ => unreachable!(),
    }

    write_huffman_desc(out, &huff_desc, weights_to_emit);
    // Jump table: 3 u16 LE giving the sizes of streams 1, 2, 3.
    out.push(s1 as u8);
    out.push((s1 >> 8) as u8);
    out.push(s2 as u8);
    out.push((s2 >> 8) as u8);
    out.push(s3 as u8);
    out.push((s3 >> 8) as u8);
    out.extend_from_slice(&streams[0]);
    out.extend_from_slice(&streams[1]);
    out.extend_from_slice(&streams[2]);
    out.extend_from_slice(&streams[3]);

    Some(start)
}

/// Huffman tree descriptor: direct (4-bit packed weights, ≤128 entries) or
/// FSE-compressed (the weight stream is itself FSE-encoded).
enum HuffDesc {
    Direct,
    FseCompressed(Vec<u8>),
}

fn write_huffman_desc(out: &mut Vec<u8>, desc: &HuffDesc, weights_to_emit: &[u8]) {
    match desc {
        HuffDesc::Direct => write_huffman_table_direct(out, weights_to_emit),
        HuffDesc::FseCompressed(bytes) => {
            // Header byte = compressed_size (< 128).
            debug_assert!(bytes.len() < 128);
            out.push(bytes.len() as u8);
            out.extend_from_slice(bytes);
        }
    }
}

fn write_huffman_table_direct(out: &mut Vec<u8>, weights_to_emit: &[u8]) {
    out.push(127 + weights_to_emit.len() as u8);
    let mut i = 0;
    while i < weights_to_emit.len() {
        let hi = weights_to_emit[i];
        let lo = if i + 1 < weights_to_emit.len() {
            weights_to_emit[i + 1]
        } else {
            0
        };
        debug_assert!(hi < 16 && lo < 16);
        out.push((hi << 4) | lo);
        i += 2;
    }
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
// Step 4: Sequence section encoder with FSE mode selection
// -------------------------------------------------------------------------

/// Which FSE compression mode to use for a sequence symbol stream.
enum SeqFseChoice {
    /// Mode 00: predefined FSE table from RFC 8878.
    Predefined,
    /// Mode 01: single repeated symbol (RLE).
    Rle(u8),
    /// Mode 10: custom FSE table built from actual symbol frequencies.
    Custom {
        encoder: FseEncoder,
        table_desc: Vec<u8>,
    },
}

/// Estimate the total bit cost of encoding `counts` with an FSE table
/// described by `weights` at the given accuracy log.  Returns the
/// approximate number of bits the FSE bitstream would use (excluding
/// table description overhead).
fn estimate_fse_bits(counts: &[u32], weights: &[i16], accuracy_log: u32) -> u64 {
    let mut bits: u64 = 0;
    for (i, &c) in counts.iter().enumerate() {
        if c == 0 {
            continue;
        }
        let w = if i < weights.len() { weights[i] } else { 0 };
        if w == 0 {
            // Symbol not in the table — predefined can't encode this.
            return u64::MAX;
        }
        let w_eff = if w == -1 { 1u32 } else { w as u32 };
        // Average bits per symbol ≈ accuracy_log - floor(log2(weight))
        let nb = accuracy_log - (31 - w_eff.leading_zeros());
        bits += c as u64 * nb as u64;
    }
    // Initial state bits
    bits += accuracy_log as u64;
    bits
}

/// Choose the best FSE compression mode for one symbol stream.
///
/// Returns `(mode_bits, choice)` where `mode_bits` is the 2-bit mode
/// value for the Symbol_Compression_Modes byte (00/01/10).
fn choose_seq_fse_mode(
    counts: &[u32],
    num_symbols: usize,
    predefined_weights: &[i16],
    predefined_al: u32,
    max_al: u32,
) -> (u8, SeqFseChoice) {
    // Find distinct symbols with non-zero count.
    let distinct: Vec<usize> = counts.iter().enumerate()
        .filter(|(_, &c)| c > 0)
        .map(|(i, _)| i)
        .collect();

    if distinct.is_empty() {
        // No symbols — shouldn't happen, but predefined is safe.
        return (0, SeqFseChoice::Predefined);
    }

    // RLE mode: only one distinct symbol.
    if distinct.len() == 1 {
        return (1, SeqFseChoice::Rle(distinct[0] as u8));
    }

    let total: u32 = counts.iter().sum();

    // Choose accuracy log: use predefined default, scale up for large blocks.
    let al = if total >= 2048 {
        max_al.min(predefined_al + 2)
    } else if total >= 512 {
        max_al.min(predefined_al + 1)
    } else {
        predefined_al
    };

    // Trim trailing zeros from counts for normalization.
    let max_sym = counts.iter().rposition(|&c| c > 0).unwrap_or(0) + 1;
    let max_sym = max_sym.max(num_symbols.min(counts.len()));

    // Normalize and build custom table.
    let table_size = 1usize << al;
    let norm = match normalize_to_acc_log(&counts[..max_sym], table_size) {
        Some(n) => n,
        None => return (0, SeqFseChoice::Predefined),
    };

    // Serialize the FSE table description.
    let mut desc_bw = ForwardBitWriter::new();
    write_fse_table_description(&mut desc_bw, &norm, al);
    let table_desc = desc_bw.finalize_no_sentinel();

    // Build encoder from the custom table.
    let dec = match FseTable::from_weights(&norm, al) {
        Ok(t) => t,
        Err(_) => return (0, SeqFseChoice::Predefined),
    };
    let encoder = FseEncoder::from_decoder(&dec, max_sym);

    // Cost comparison: predefined vs custom.
    let predefined_bits = estimate_fse_bits(counts, predefined_weights, predefined_al);
    let custom_bits = estimate_fse_bits(&counts[..max_sym], &norm, al);
    let custom_total = custom_bits.saturating_add(table_desc.len() as u64 * 8);

    if custom_total < predefined_bits {
        (2, SeqFseChoice::Custom { encoder, table_desc })
    } else {
        (0, SeqFseChoice::Predefined)
    }
}

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

    // Count symbol frequencies for each stream.
    let mut ll_freq = [0u32; 36];
    let mut of_freq = [0u32; 32];
    let mut ml_freq = [0u32; 53];
    for (of_c, ml_c, ll_c) in &codes {
        ll_freq[ll_c.code as usize] += 1;
        of_freq[of_c.code as usize] += 1;
        ml_freq[ml_c.code as usize] += 1;
    }

    // Choose best FSE mode for each stream.
    let (ll_mode, ll_choice) = choose_seq_fse_mode(
        &ll_freq, 36, &PREDEFINED_LL_WEIGHTS, 6, 9,
    );
    let (of_mode, of_choice) = choose_seq_fse_mode(
        &of_freq, 32, &PREDEFINED_OF_WEIGHTS, 5, 8,
    );
    let (ml_mode, ml_choice) = choose_seq_fse_mode(
        &ml_freq, 53, &PREDEFINED_ML_WEIGHTS, 6, 9,
    );

    // Symbol_Compression_Modes byte: [LL_Mode:2][OF_Mode:2][ML_Mode:2][Reserved:2]
    let mode_byte = (ll_mode << 6) | (of_mode << 4) | (ml_mode << 2);
    out.push(mode_byte);

    // Write table descriptions / RLE symbols in order: LL, OF, ML.
    for choice in [&ll_choice, &of_choice, &ml_choice] {
        match choice {
            SeqFseChoice::Rle(sym) => out.push(*sym),
            SeqFseChoice::Custom { table_desc, .. } => out.extend_from_slice(table_desc),
            SeqFseChoice::Predefined => {}
        }
    }

    // Get the actual encoder references for the bitstream.
    // If any stream uses RLE, we need to build temporary encoders.
    let has_rle = matches!(ll_choice, SeqFseChoice::Rle(_))
        || matches!(of_choice, SeqFseChoice::Rle(_))
        || matches!(ml_choice, SeqFseChoice::Rle(_));

    if has_rle {
        return encode_sequences_with_rle(out, &codes, n,
            &ll_choice, &of_choice, &ml_choice);
    }

    let ll_enc: &FseEncoder = match &ll_choice {
        SeqFseChoice::Predefined => &PREDEFINED_LL_ENC,
        SeqFseChoice::Custom { encoder, .. } => encoder,
        SeqFseChoice::Rle(_) => unreachable!(),
    };
    let of_enc: &FseEncoder = match &of_choice {
        SeqFseChoice::Predefined => &PREDEFINED_OF_ENC,
        SeqFseChoice::Custom { encoder, .. } => encoder,
        SeqFseChoice::Rle(_) => unreachable!(),
    };
    let ml_enc: &FseEncoder = match &ml_choice {
        SeqFseChoice::Predefined => &PREDEFINED_ML_ENC,
        SeqFseChoice::Custom { encoder, .. } => encoder,
        SeqFseChoice::Rle(_) => unreachable!(),
    };

    encode_bitstream(&mut out, &codes, n, ll_enc, of_enc, ml_enc);
    Some(out)
}

/// Build an FseEncoder for a single RLE symbol.
fn make_rle_encoder(sym: u8, num_symbols: usize) -> FseEncoder {
    let dec = fse::fse_rle_table(sym);
    FseEncoder::from_decoder(&dec, num_symbols)
}

/// Fallback path when any stream uses RLE mode — we can't borrow the RLE
/// encoder from the enum variant, so we construct temporary encoders here.
fn encode_sequences_with_rle(
    mut out: Vec<u8>,
    codes: &[(SeqCodes, SeqCodes, SeqCodes)],
    n: usize,
    ll_choice: &SeqFseChoice,
    of_choice: &SeqFseChoice,
    ml_choice: &SeqFseChoice,
) -> Option<Vec<u8>> {
    let ll_rle;
    let of_rle;
    let ml_rle;
    let ll_enc: &FseEncoder = match ll_choice {
        SeqFseChoice::Predefined => &PREDEFINED_LL_ENC,
        SeqFseChoice::Custom { encoder, .. } => encoder,
        SeqFseChoice::Rle(sym) => { ll_rle = make_rle_encoder(*sym, 36); &ll_rle }
    };
    let of_enc: &FseEncoder = match of_choice {
        SeqFseChoice::Predefined => &PREDEFINED_OF_ENC,
        SeqFseChoice::Custom { encoder, .. } => encoder,
        SeqFseChoice::Rle(sym) => { of_rle = make_rle_encoder(*sym, 32); &of_rle }
    };
    let ml_enc: &FseEncoder = match ml_choice {
        SeqFseChoice::Predefined => &PREDEFINED_ML_ENC,
        SeqFseChoice::Custom { encoder, .. } => encoder,
        SeqFseChoice::Rle(sym) => { ml_rle = make_rle_encoder(*sym, 53); &ml_rle }
    };
    encode_bitstream(&mut out, codes, n, ll_enc, of_enc, ml_enc);
    Some(out)
}

/// Encode the FSE bitstream for sequences using the given encoder tables.
fn encode_bitstream(
    out: &mut Vec<u8>,
    codes: &[(SeqCodes, SeqCodes, SeqCodes)],
    n: usize,
    ll_enc: &FseEncoder,
    of_enc: &FseEncoder,
    ml_enc: &FseEncoder,
) {
    let mut bw = ForwardBitWriter::with_capacity(n * 4);

    // Initial encoder states pinned to the LAST sequence's symbols.
    let last = codes.last().unwrap();
    let mut state_of = of_enc.start_state(last.0.code);
    let mut state_ml = ml_enc.start_state(last.1.code);
    let mut state_ll = ll_enc.start_state(last.2.code);

    // Last sequence in encoder time = first encoded.  Only its EXTRA bits go
    // out (no preceding state-update bits).
    write_extra_lmo(&mut bw, &last.2, &last.1, &last.0);

    // Walk all preceding sequences in REVERSE encoder time.
    for i in (0..n - 1).rev() {
        let (of_c, ml_c, ll_c) = &codes[i];
        state_of = of_enc.encode_symbol(state_of, of_c.code, &mut bw);
        state_ml = ml_enc.encode_symbol(state_ml, ml_c.code, &mut bw);
        state_ll = ll_enc.encode_symbol(state_ll, ll_c.code, &mut bw);
        write_extra_lmo(&mut bw, ll_c, ml_c, of_c);
    }

    // Initial states.  Decoder reads ll → of → ml; encoder writes ml → of → ll.
    bw.write_bits(state_ml as u64, ml_enc.accuracy_log);
    bw.write_bits(state_of as u64, of_enc.accuracy_log);
    bw.write_bits(state_ll as u64, ll_enc.accuracy_log);

    let bitstream = bw.finalize();
    out.extend_from_slice(&bitstream);
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
