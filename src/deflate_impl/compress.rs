//! DEFLATE compression (RFC 1951)
//!
//! Implements LZ77 string matching with hash chains and Huffman coding.
//! Supports compression levels 0 (stored only) through 9 (maximum compression).

use super::bitwriter::BitWriter;
use super::huffman;
use super::tables;

const WINDOW_SIZE: usize = 32768;
const WINDOW_MASK: usize = WINDOW_SIZE - 1;
const HASH_BITS: usize = 15;
const HASH_SIZE: usize = 1 << HASH_BITS;
const HASH_MASK: usize = HASH_SIZE - 1;
const MIN_MATCH: usize = 3;
const MAX_MATCH: usize = 258;
const MAX_STORED_BLOCK: usize = 65535;
const BLOCK_SIZE: usize = 32768;

const NONE: u32 = u32::MAX; // sentinel for empty hash chain

// ---------------------------------------------------------------------------
// Compression level configuration (inspired by zlib)
// ---------------------------------------------------------------------------

struct Config {
    good_length: usize,
    nice_length: usize,
    max_chain: usize,
    insert_step: usize, // 0 = skip all match insertions, 1 = every pos, N = every Nth
    use_fixed: bool,    // true = use fixed Huffman codes (skip tree building)
}

const CONFIGS: [Config; 10] = [
    Config { good_length: 0, nice_length: 0, max_chain: 0, insert_step: 0, use_fixed: false },        // 0: stored
    Config { good_length: 4, nice_length: 8, max_chain: 4, insert_step: 1, use_fixed: true },         // 1
    Config { good_length: 4, nice_length: 16, max_chain: 8, insert_step: 4, use_fixed: true },        // 2
    Config { good_length: 4, nice_length: 32, max_chain: 32, insert_step: 4, use_fixed: true },       // 3
    Config { good_length: 4, nice_length: 16, max_chain: 16, insert_step: 2, use_fixed: false },      // 4
    Config { good_length: 8, nice_length: 32, max_chain: 32, insert_step: 1, use_fixed: false },      // 5
    Config { good_length: 8, nice_length: 128, max_chain: 128, insert_step: 1, use_fixed: false },    // 6: default
    Config { good_length: 8, nice_length: 128, max_chain: 256, insert_step: 1, use_fixed: false },    // 7
    Config { good_length: 32, nice_length: 258, max_chain: 1024, insert_step: 1, use_fixed: false },  // 8
    Config { good_length: 32, nice_length: 258, max_chain: 4096, insert_step: 1, use_fixed: false },  // 9
];

// ---------------------------------------------------------------------------
// Token representation
// ---------------------------------------------------------------------------

enum Token {
    Literal(u8),
    Match { length: u16, distance: u16 },
}

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

/// Compress `input` into a raw DEFLATE stream.
pub fn deflate(input: &[u8], level: u32) -> Vec<u8> {
    let level = std::cmp::min(level, 9) as usize;

    if level == 0 || input.is_empty() {
        return compress_stored_all(input);
    }

    compress_with_huffman(input, level)
}

// ---------------------------------------------------------------------------
// Stored-block compression (level 0 / fallback)
// ---------------------------------------------------------------------------

fn compress_stored_all(input: &[u8]) -> Vec<u8> {
    let mut w = BitWriter::with_capacity(input.len() + input.len() / MAX_STORED_BLOCK * 5 + 20);
    let mut offset = 0;

    if input.is_empty() {
        write_stored_block(&mut w, &[], true);
        return w.finish();
    }

    while offset < input.len() {
        let chunk = std::cmp::min(MAX_STORED_BLOCK, input.len() - offset);
        let is_final = offset + chunk >= input.len();
        write_stored_block(&mut w, &input[offset..offset + chunk], is_final);
        offset += chunk;
    }
    w.finish()
}

fn write_stored_block(w: &mut BitWriter, data: &[u8], is_final: bool) {
    w.write_bits(is_final as u32, 1);
    w.write_bits(0b00, 2); // BTYPE = stored
    w.align_to_byte();
    let len = data.len() as u16;
    w.write_u16_le(len);
    w.write_u16_le(!len);
    w.write_bytes(data);
}

// ---------------------------------------------------------------------------
// Huffman-block compression (levels 1-9)
// ---------------------------------------------------------------------------

fn compress_with_huffman(input: &[u8], level: usize) -> Vec<u8> {
    let config = &CONFIGS[level];
    let mut w = BitWriter::with_capacity(input.len());

    let mut offset = 0;
    while offset < input.len() {
        let end = std::cmp::min(offset + BLOCK_SIZE, input.len());
        let is_final = end >= input.len();

        let tokens = lz77(input, offset, end, config);
        write_best_block(&mut w, &input[offset..end], &tokens, is_final, config.use_fixed);

        offset = end;
    }
    w.finish()
}

/// Choose the smallest block encoding and write it.
fn write_best_block(
    w: &mut BitWriter,
    raw_data: &[u8],
    tokens: &[Token],
    is_final: bool,
    use_fixed: bool,
) {
    let stored_bits = stored_block_bits(raw_data.len());

    if use_fixed {
        // Fast path for low levels: compare stored vs fixed Huffman only.
        let fixed_bits = estimate_fixed_bits(tokens);
        if stored_bits <= fixed_bits && raw_data.len() <= MAX_STORED_BLOCK {
            write_stored_block(w, raw_data, is_final);
        } else {
            write_fixed_block(w, tokens, is_final);
        }
    } else {
        // Full path: compare stored vs dynamic Huffman.
        let mut lit_freq = [0u32; 286];
        let mut dist_freq = [0u32; 30];
        lit_freq[256] = 1;
        for token in tokens {
            match token {
                Token::Literal(b) => lit_freq[*b as usize] += 1,
                Token::Match { length, distance } => {
                    let (sym, _, _) = tables::length_to_symbol(*length);
                    lit_freq[sym as usize] += 1;
                    let (dsym, _, _) = tables::distance_to_symbol(*distance);
                    dist_freq[dsym as usize] += 1;
                }
            }
        }
        let lit_lengths = huffman::build_lengths(&lit_freq, 15);
        let dist_lengths = huffman::build_lengths(&dist_freq, 15);
        let dynamic_bits = estimate_dynamic_bits(tokens, &lit_lengths, &dist_lengths, &lit_freq, &dist_freq);

        if stored_bits <= dynamic_bits && raw_data.len() <= MAX_STORED_BLOCK {
            write_stored_block(w, raw_data, is_final);
        } else {
            write_dynamic_block(w, tokens, &lit_lengths, &dist_lengths, is_final);
        }
    }
}

fn stored_block_bits(data_len: usize) -> usize {
    // 3 bits header + align (worst 7) + 4 bytes len/nlen + data
    3 + 7 + 32 + data_len * 8
}

fn estimate_fixed_bits(tokens: &[Token]) -> usize {
    static FIXED_LIT: std::sync::OnceLock<[u8; 288]> = std::sync::OnceLock::new();
    static FIXED_DIST: std::sync::OnceLock<[u8; 32]> = std::sync::OnceLock::new();
    let fl = FIXED_LIT.get_or_init(tables::fixed_literal_lengths);
    let fd = FIXED_DIST.get_or_init(tables::fixed_distance_lengths);

    let mut bits = 3usize; // block header
    for token in tokens {
        match token {
            Token::Literal(b) => bits += fl[*b as usize] as usize,
            Token::Match { length, distance } => {
                let (sym, extra, _) = tables::length_to_symbol(*length);
                bits += fl[sym as usize] as usize + extra as usize;
                let (dsym, dextra, _) = tables::distance_to_symbol(*distance);
                bits += fd[dsym as usize] as usize + dextra as usize;
            }
        }
    }
    bits += fl[256] as usize; // end-of-block
    bits
}

// ---------------------------------------------------------------------------
// Fixed Huffman block writer (levels 1-3)
// ---------------------------------------------------------------------------

fn fixed_lit_codes() -> &'static [(u32, u8)] {
    static CODES: std::sync::OnceLock<Vec<(u32, u8)>> = std::sync::OnceLock::new();
    CODES.get_or_init(|| huffman::canonical_codes(&tables::fixed_literal_lengths()))
}

fn fixed_dist_codes() -> &'static [(u32, u8)] {
    static CODES: std::sync::OnceLock<Vec<(u32, u8)>> = std::sync::OnceLock::new();
    CODES.get_or_init(|| huffman::canonical_codes(&tables::fixed_distance_lengths()))
}

fn write_fixed_block(w: &mut BitWriter, tokens: &[Token], is_final: bool) {
    w.write_bits(is_final as u32, 1);
    w.write_bits(0b01, 2); // BTYPE = fixed Huffman

    let lit_codes = fixed_lit_codes();
    let dist_codes = fixed_dist_codes();

    for token in tokens {
        match token {
            Token::Literal(b) => {
                let (code, len) = lit_codes[*b as usize];
                w.write_bits(code, len as u32);
            }
            Token::Match { length, distance } => {
                let (sym, extra_bits, extra_val) = tables::length_to_symbol(*length);
                let (code, len) = lit_codes[sym as usize];
                w.write_bits(code, len as u32);
                if extra_bits > 0 {
                    w.write_bits(extra_val as u32, extra_bits as u32);
                }
                let (dsym, dextra_bits, dextra_val) = tables::distance_to_symbol(*distance);
                let (dcode, dlen) = dist_codes[dsym as usize];
                w.write_bits(dcode, dlen as u32);
                if dextra_bits > 0 {
                    w.write_bits(dextra_val as u32, dextra_bits as u32);
                }
            }
        }
    }

    // End-of-block symbol (256).
    let (code, len) = lit_codes[256];
    w.write_bits(code, len as u32);
}

fn estimate_dynamic_bits(
    tokens: &[Token],
    lit_lengths: &[u8],
    dist_lengths: &[u8],
    _lit_freq: &[u32],
    _dist_freq: &[u32],
) -> usize {
    // 3 bits block header + header overhead (generous estimate) + data bits
    let header_est = 3 + 5 + 5 + 4 + 19 * 3 + 286 * 4 + 30 * 4; // rough upper bound

    let mut data_bits = 0usize;
    for token in tokens {
        match token {
            Token::Literal(b) => data_bits += lit_lengths[*b as usize] as usize,
            Token::Match { length, distance } => {
                let (sym, extra, _) = tables::length_to_symbol(*length);
                data_bits += lit_lengths[sym as usize] as usize + extra as usize;
                let (dsym, dextra, _) = tables::distance_to_symbol(*distance);
                data_bits += dist_lengths[dsym as usize] as usize + dextra as usize;
            }
        }
    }
    // End-of-block symbol.
    data_bits += lit_lengths[256] as usize;

    header_est + data_bits
}

// ---------------------------------------------------------------------------
// Dynamic Huffman block writer
// ---------------------------------------------------------------------------

fn write_dynamic_block(
    w: &mut BitWriter,
    tokens: &[Token],
    lit_lengths: &[u8],
    dist_lengths: &[u8],
    is_final: bool,
) {
    w.write_bits(is_final as u32, 1);
    w.write_bits(0b10, 2); // BTYPE = dynamic Huffman

    // Trim trailing zeros.
    let mut hlit = lit_lengths.len();
    while hlit > 257 && lit_lengths[hlit - 1] == 0 {
        hlit -= 1;
    }

    // RFC 1951 requires the dist alphabet to encode at least one valid code.
    // When a block has zero matches every dist length is 0 — that produces
    // a degenerate canonical Huffman table that strict decoders (zlib,
    // flate2) reject as "corrupt deflate stream", though our own decoder
    // happens to accept it.  zlib's encoder sidesteps this by emitting two
    // synthetic 1-bit dist codes that never get used in the bitstream
    // (`zlib/trees.c::send_all_trees`).  Mirror that here: copy
    // `dist_lengths` into a local buffer and patch positions 0 and 1 to 1
    // when the original is all-zero.
    let dist_owned: Vec<u8>;
    let dist_lengths: &[u8] = if dist_lengths.iter().all(|&l| l == 0) {
        let mut buf = vec![0u8; dist_lengths.len().max(2)];
        buf[0] = 1;
        buf[1] = 1;
        dist_owned = buf;
        &dist_owned
    } else {
        dist_lengths
    };

    let mut hdist = dist_lengths.len();
    while hdist > 1 && dist_lengths[hdist - 1] == 0 {
        hdist -= 1;
    }


    // RLE-encode the combined code-length sequence.
    let mut combined: Vec<u8> = Vec::with_capacity(hlit + hdist);
    combined.extend_from_slice(&lit_lengths[..hlit]);
    combined.extend_from_slice(&dist_lengths[..hdist]);
    let rle = rle_encode(&combined);

    // Build Huffman codes for the code-length alphabet (max 7 bits).
    let mut cl_freq = [0u32; 19];
    for &(sym, _) in &rle {
        cl_freq[sym as usize] += 1;
    }
    let mut cl_lengths_arr = huffman::build_lengths(&cl_freq, 7);
    // Strict inflaters require the code-length code itself to be complete
    // even when only one symbol is used (miniz_oxide: `bt == HUFFLEN_TABLE`).
    // A lone 1-bit code is incomplete, so pair it with an unused symbol.
    if cl_lengths_arr.iter().filter(|&&l| l > 0).count() == 1 {
        let dummy = (0..19).find(|&i| cl_lengths_arr[i] == 0).expect("19-symbol alphabet");
        cl_lengths_arr[dummy] = 1;
    }
    let cl_codes = huffman::canonical_codes(&cl_lengths_arr);

    // Determine HCLEN.
    let mut hclen = 19;
    while hclen > 4 && cl_lengths_arr[tables::CODE_LENGTH_ORDER[hclen - 1]] == 0 {
        hclen -= 1;
    }

    // Write header.
    w.write_bits((hlit - 257) as u32, 5);
    w.write_bits((hdist - 1) as u32, 5);
    w.write_bits((hclen - 4) as u32, 4);

    for i in 0..hclen {
        w.write_bits(cl_lengths_arr[tables::CODE_LENGTH_ORDER[i]] as u32, 3);
    }

    // Write the RLE-encoded code lengths.
    for &(sym, extra_val) in &rle {
        let (code, len) = cl_codes[sym as usize];
        w.write_bits(code, len as u32);
        match sym {
            16 => w.write_bits(extra_val as u32, 2),
            17 => w.write_bits(extra_val as u32, 3),
            18 => w.write_bits(extra_val as u32, 7),
            _ => {}
        }
    }

    // Encode the actual data.
    let lit_codes = huffman::canonical_codes(lit_lengths);
    let dist_codes = huffman::canonical_codes(dist_lengths);

    for token in tokens {
        match token {
            Token::Literal(b) => {
                huffman::encode_symbol(w, &lit_codes, *b as u16);
            }
            Token::Match { length, distance } => {
                let (sym, extra_bits, extra_val) = tables::length_to_symbol(*length);
                huffman::encode_symbol(w, &lit_codes, sym);
                if extra_bits > 0 {
                    w.write_bits(extra_val as u32, extra_bits as u32);
                }
                let (dsym, dextra_bits, dextra_val) = tables::distance_to_symbol(*distance);
                huffman::encode_symbol(w, &dist_codes, dsym as u16);
                if dextra_bits > 0 {
                    w.write_bits(dextra_val as u32, dextra_bits as u32);
                }
            }
        }
    }

    // End-of-block.
    huffman::encode_symbol(w, &lit_codes, 256);
}

/// Run-length encode a sequence of code lengths.
///
/// Returns `(symbol, extra_bits_value)` pairs.
fn rle_encode(lengths: &[u8]) -> Vec<(u8, u8)> {
    let mut result = Vec::new();
    let mut i = 0;

    while i < lengths.len() {
        let val = lengths[i];
        let mut run = 1;
        while i + run < lengths.len() && lengths[i + run] == val {
            run += 1;
        }

        if val == 0 {
            let mut remaining = run;
            while remaining > 0 {
                if remaining >= 11 {
                    let n = std::cmp::min(remaining, 138);
                    result.push((18u8, (n - 11) as u8));
                    remaining -= n;
                } else if remaining >= 3 {
                    let n = std::cmp::min(remaining, 10);
                    result.push((17u8, (n - 3) as u8));
                    remaining -= n;
                } else {
                    result.push((0u8, 0));
                    remaining -= 1;
                }
            }
        } else {
            // Emit the value once.
            result.push((val, 0));
            let mut remaining = run - 1;
            while remaining > 0 {
                if remaining >= 3 {
                    let n = std::cmp::min(remaining, 6);
                    result.push((16u8, (n - 3) as u8));
                    remaining -= n;
                } else {
                    result.push((val, 0));
                    remaining -= 1;
                }
            }
        }

        i += run;
    }

    result
}

// ---------------------------------------------------------------------------
// LZ77 string matching
// ---------------------------------------------------------------------------

fn hash3(data: &[u8]) -> usize {
    let v = (data[0] as u32) | ((data[1] as u32) << 8) | ((data[2] as u32) << 16);
    (v.wrapping_mul(0x1E35_A7BD) >> (32 - HASH_BITS)) as usize & HASH_MASK
}

fn lz77(input: &[u8], start: usize, end: usize, config: &Config) -> Vec<Token> {
    let mut tokens = Vec::with_capacity(end - start);
    let mut head = vec![NONE; HASH_SIZE];
    let mut prev = vec![NONE; WINDOW_SIZE];
    let mut pos = start;

    // Pre-populate hash chains from before `start` (provides context for the window).
    let pre_start = start.saturating_sub(WINDOW_SIZE);
    for p in pre_start..start {
        if p + MIN_MATCH <= end {
            let h = hash3(&input[p..]);
            prev[p & WINDOW_MASK] = head[h];
            head[h] = p as u32;
        }
    }

    while pos < end {
        let remaining = end - pos;
        if remaining < MIN_MATCH {
            tokens.push(Token::Literal(input[pos]));
            pos += 1;
            continue;
        }

        let h = hash3(&input[pos..]);

        // Find best match by walking the hash chain.
        let mut best_len = MIN_MATCH - 1;
        let mut best_dist = 0u16;
        let min_pos = pos.saturating_sub(WINDOW_SIZE);
        let mut chain = config.max_chain;
        let mut match_head = head[h];

        while match_head != NONE && chain > 0 {
            let mp = match_head as usize;
            if mp < min_pos || mp >= pos {
                break;
            }
            let dist = pos - mp;

            // Quick check: first and last bytes of current best.
            if mp + best_len < input.len()
                && pos + best_len < input.len()
                && input[mp + best_len] == input[pos + best_len]
                && input[mp] == input[pos]
            {
                let max_len = std::cmp::min(MAX_MATCH, remaining);
                let mut len = 0;
                while len < max_len && input[mp + len] == input[pos + len] {
                    len += 1;
                }
                if len > best_len {
                    best_len = len;
                    best_dist = dist as u16;
                    if best_len >= config.nice_length {
                        break;
                    }
                }
            }

            let next = prev[mp & WINDOW_MASK];
            if next == NONE || next as usize >= mp {
                break; // chain went forward / ended
            }
            match_head = next;
            if best_len >= config.good_length {
                chain >>= 2; // reduce search effort for "good enough" matches
            }
            chain = chain.saturating_sub(1);
        }

        // Update hash chain.
        prev[pos & WINDOW_MASK] = head[h];
        head[h] = pos as u32;

        if best_len >= MIN_MATCH {
            tokens.push(Token::Match {
                length: best_len as u16,
                distance: best_dist,
            });
            // Insert hash entries for positions inside the match.
            // At low levels, skip most/all insertions for speed.
            if config.insert_step > 0 {
                let mut i = config.insert_step;
                while i < best_len {
                    let p = pos + i;
                    if p + MIN_MATCH <= end {
                        let h2 = hash3(&input[p..]);
                        prev[p & WINDOW_MASK] = head[h2];
                        head[h2] = p as u32;
                    }
                    i += config.insert_step;
                }
            }
            pos += best_len;
        } else {
            tokens.push(Token::Literal(input[pos]));
            pos += 1;
        }
    }

    tokens
}
