//! bzip2 encoder.
//!
//! Pipeline (per block):
//!   1. RLE1   — runs of 4..=255 same bytes → `bbbb<count>`.
//!   2. BWT    — sort all rotations, output the last column + origin.
//!   3. MTF    — move-to-front on the BWT output (mapped to alphabet indices).
//!   4. RLE2   — runs of zeros → RUNA/RUNB sequences.
//!   5. Huffman — multi-table canonical Huffman with selector groups.
//!   6. Frame  — 4-byte file header, per-block bit-packed payload, EOS marker.
//!
//! This is an MVP encoder.  It uses a simple `Vec<&[u8]>::sort_by` for the
//! BWT instead of a real suffix array, and starts with 2 Huffman tables and
//! one refinement pass instead of bzip2's full 4-pass selector optimization.
//! Both can be improved later.

use std::cmp::Ordering;

use super::bits::BitWriter;
use super::crc::Crc32;

const BLOCK_MAGIC: u64 = 0x3141_5926_5359;
const EOS_MAGIC: u64 = 0x1772_4538_5090;
const HUFFMAN_GROUP_SIZE: usize = 50;
const MAX_HUFFMAN_TABLES: usize = 6;
const MAX_HUFFMAN_CODE_LEN: u8 = 17;
const MAX_SELECTORS: usize = 18002;

/// Encode `input` as a complete bzip2 stream at the given level (1..=9).
pub fn encode_stream(input: &[u8], level: u32) -> Vec<u8> {
    let level = level.clamp(1, 9);
    let block_size = (level as usize) * 100_000;

    let mut bw = BitWriter::new();
    // File header: BZh<level>
    bw.write_bits(b'B' as u64, 8);
    bw.write_bits(b'Z' as u64, 8);
    bw.write_bits(b'h' as u64, 8);
    bw.write_bits((b'0' + level as u8) as u64, 8);

    let mut combined_crc: u32 = 0;
    let mut pos = 0usize;
    while pos < input.len() {
        // Pre-RLE1, the block input is at most block_size bytes.  In a smarter
        // encoder we'd target a post-RLE1 size of block_size; this MVP just
        // splits the raw input.
        let take = (input.len() - pos).min(block_size);
        let block = &input[pos..pos + take];
        pos += take;

        let block_crc = compute_block_crc(block);
        combined_crc = combined_crc.rotate_left(1) ^ block_crc;

        encode_block(&mut bw, block, block_crc);
    }

    // End-of-stream marker.
    bw.write_bits(EOS_MAGIC, 48);
    bw.write_bits(combined_crc as u64, 32);
    bw.align_to_byte();
    bw.finish()
}

fn compute_block_crc(block: &[u8]) -> u32 {
    let mut c = Crc32::new();
    c.update(block);
    c.finalize()
}

fn encode_block(bw: &mut BitWriter, block: &[u8], block_crc: u32) {
    // -- 1. RLE1 --
    let rle1_out = forward_rle1(block);

    // -- 2. BWT --
    let (bwt_out, origin) = forward_bwt(&rle1_out);

    // -- 3. MTF + 4. RLE2 (combined; we don't need an intermediate buffer) --
    // Compute the symbol-used bitmap so the MTF list initialization matches
    // what the decoder will reconstruct.
    let mut symbol_used = [false; 256];
    for &b in &bwt_out {
        symbol_used[b as usize] = true;
    }
    let num_used: usize = symbol_used.iter().filter(|&&u| u).count();
    debug_assert!(num_used >= 1);

    // Build byte → alphabet-index map.
    let mut byte_to_alpha = [0u16; 256];
    let mut alphabet_to_byte = vec![0u8; num_used];
    {
        let mut idx = 0u16;
        for (b, used) in symbol_used.iter().enumerate() {
            if *used {
                byte_to_alpha[b] = idx;
                alphabet_to_byte[idx as usize] = b as u8;
                idx += 1;
            }
        }
    }

    // MTF list of alphabet indices.
    let mut mtf: Vec<u16> = (0..num_used as u16).collect();
    let alpha_size = num_used + 2; // RUNA + RUNB + (num_used - 1) MTF symbols + EOB
    let eob_symbol = (num_used + 1) as u16;

    // Output: a sequence of u16 alphabet symbols (RUNA=0, RUNB=1, MTF1..=num_used,
    // EOB=num_used+1).
    let mut symbols: Vec<u16> = Vec::with_capacity(bwt_out.len());
    let mut zero_run: u32 = 0;

    for &b in &bwt_out {
        let alpha_idx = byte_to_alpha[b as usize];
        // Find the MTF position of this alphabet index.
        let pos = mtf.iter().position(|&x| x == alpha_idx).unwrap();
        // Move to front.
        for i in (1..=pos).rev() {
            mtf[i] = mtf[i - 1];
        }
        mtf[0] = alpha_idx;

        if pos == 0 {
            zero_run += 1;
        } else {
            // Flush any pending zero run as RUNA/RUNB.
            if zero_run > 0 {
                emit_zero_run(&mut symbols, zero_run);
                zero_run = 0;
            }
            // Real symbol: pos in 1..num_used → alphabet symbol pos+1.
            symbols.push((pos + 1) as u16);
        }
    }
    if zero_run > 0 {
        emit_zero_run(&mut symbols, zero_run);
    }
    symbols.push(eob_symbol);

    // -- 5. Huffman: build tables and selectors --
    let (tables, selectors) = build_huffman_tables(&symbols, alpha_size);

    // -- 6. Write the block header --
    bw.write_bits(BLOCK_MAGIC, 48);
    bw.write_bits(block_crc as u64, 32);
    bw.write_bits(0, 1); // randomized = 0
    bw.write_bits(origin as u64, 24);

    // Symbol-used bitmap.
    let mut high: u32 = 0;
    let mut lows: [u32; 16] = [0; 16];
    for (b, used) in symbol_used.iter().enumerate() {
        if *used {
            let h = b / 16;
            let l = b % 16;
            high |= 1 << (15 - h);
            lows[h] |= 1 << (15 - l);
        }
    }
    bw.write_bits(high as u64, 16);
    for h in 0..16 {
        if (high >> (15 - h)) & 1 != 0 {
            bw.write_bits(lows[h] as u64, 16);
        }
    }

    // Number of Huffman tables.
    let num_tables = tables.len();
    debug_assert!((2..=MAX_HUFFMAN_TABLES).contains(&num_tables));
    bw.write_bits(num_tables as u64, 3);

    // Selectors: MTF-encode the selector list, then write each as unary.
    bw.write_bits(selectors.len() as u64, 15);
    let mut sel_pos: [u8; MAX_HUFFMAN_TABLES] = [0, 1, 2, 3, 4, 5];
    for &s in &selectors {
        let s = s as usize;
        let pos = sel_pos.iter().position(|&p| p == s as u8).unwrap();
        // Move to front.
        for i in (1..=pos).rev() {
            sel_pos[i] = sel_pos[i - 1];
        }
        sel_pos[0] = s as u8;
        // Write `pos` ones followed by a zero.
        for _ in 0..pos {
            bw.write_bits(1, 1);
        }
        bw.write_bits(0, 1);
    }

    // Per-table code lengths: 5-bit start, then deltas.
    for table in &tables {
        let mut current = table.code_lens[0] as i32;
        bw.write_bits(current as u64, 5);
        for &len in &table.code_lens {
            let target = len as i32;
            while current != target {
                if current < target {
                    bw.write_bits(1, 1); // change
                    bw.write_bits(0, 1); // increase
                    current += 1;
                } else {
                    bw.write_bits(1, 1);
                    bw.write_bits(1, 1); // decrease
                    current -= 1;
                }
            }
            bw.write_bits(0, 1); // no more change
        }
    }

    // Encoded data: write each symbol with the table chosen by its 50-symbol group.
    for (i, &sym) in symbols.iter().enumerate() {
        let group = i / HUFFMAN_GROUP_SIZE;
        let table = &tables[selectors[group] as usize];
        bw.write_bits(table.codes[sym as usize] as u64, table.code_lens[sym as usize] as u32);
    }
}

fn emit_zero_run(symbols: &mut Vec<u16>, mut n: u32) {
    // RUNA = 0, RUNB = 1.  Encoding: subtract 1, then read bits LSB-first;
    // bit 0 → RUNA, bit 1 → RUNB.  Equivalently: while n > 0 { emit RUNA if
    // (n & 1) == 1 else RUNB; n = (n - 1) >> 1; n -= 0 if RUNA else 1 } ...
    // Simpler closed form:
    while n > 0 {
        n -= 1;
        symbols.push((n & 1) as u16);
        n >>= 1;
    }
}

// =========================================================================
// Forward RLE1
// =========================================================================
fn forward_rle1(input: &[u8]) -> Vec<u8> {
    let mut out = Vec::with_capacity(input.len() + input.len() / 8);
    let mut i = 0usize;
    while i < input.len() {
        let b = input[i];
        let mut run = 1usize;
        while i + run < input.len() && input[i + run] == b && run < 255 {
            run += 1;
        }
        if run >= 4 {
            out.push(b);
            out.push(b);
            out.push(b);
            out.push(b);
            out.push((run - 4) as u8);
        } else {
            for _ in 0..run {
                out.push(b);
            }
        }
        i += run;
    }
    out
}

// =========================================================================
// Forward BWT
// =========================================================================
//
// We sort all `n` rotations of the input.  Naive comparator-based sorting
// is O(n² log n) because each rotation comparison can scan up to n bytes.
// Instead we use Manber-Myers prefix doubling on the DOUBLED input
// (`input ++ input`): rotations of `input` correspond to length-`n`
// prefixes of suffixes 0..n of the doubled input.  Manber-Myers sorts
// suffixes by progressively longer prefixes (1, 2, 4, 8, ...), reaching a
// total complexity of O(n log² n) which is fast enough for the 900 KiB
// max block size.
fn forward_bwt(input: &[u8]) -> (Vec<u8>, usize) {
    let n = input.len();
    if n == 0 {
        return (Vec::new(), 0);
    }
    if n == 1 {
        return (vec![input[0]], 0);
    }

    // Sort the n rotation starts using prefix doubling.
    let order = manber_myers_rotations(input);

    // Last column: for each row in sorted order, take input[(start + n - 1) % n].
    let mut last = Vec::with_capacity(n);
    let mut origin = 0usize;
    for (row, &start) in order.iter().enumerate() {
        let last_byte_index = (start as usize + n - 1) % n;
        last.push(input[last_byte_index]);
        if start == 0 {
            origin = row;
        }
    }
    (last, origin)
}

/// Manber-Myers prefix doubling on rotations of `input`.  Returns a vector
/// `order[i] = the starting index (0..n) of the i-th rotation in sorted
/// lexicographic order`.
///
/// Implementation note: rotations are compared in modular space, so the
/// "second key" at offset `k` wraps around using `(pos + k) % n`.
fn manber_myers_rotations(input: &[u8]) -> Vec<u32> {
    let n = input.len();
    let mut order: Vec<u32> = (0..n as u32).collect();
    let mut rank: Vec<u32> = input.iter().map(|&b| b as u32).collect();
    let mut new_rank = vec![0u32; n];

    let mut k = 1usize;
    loop {
        // Sort `order` by (rank[a], rank[(a+k) % n]) — fall back to a stable
        // tie-breaker by index so equal pairs end up in a deterministic order.
        order.sort_unstable_by(|&a, &b| {
            let ra = rank[a as usize];
            let rb = rank[b as usize];
            if ra != rb {
                return ra.cmp(&rb);
            }
            let na = (a as usize + k) % n;
            let nb = (b as usize + k) % n;
            rank[na].cmp(&rank[nb])
        });

        // Reassign ranks based on the new order.
        new_rank[order[0] as usize] = 0;
        for i in 1..n {
            let prev = order[i - 1] as usize;
            let cur = order[i] as usize;
            let same = rank[prev] == rank[cur]
                && rank[(prev + k) % n] == rank[(cur + k) % n];
            new_rank[cur] = new_rank[prev] + if same { 0 } else { 1 };
        }

        // If all ranks are unique we're done — every rotation has been
        // distinguished from every other.
        if new_rank[order[n - 1] as usize] as usize == n - 1 {
            return order;
        }
        std::mem::swap(&mut rank, &mut new_rank);
        k *= 2;
        if k >= n {
            return order;
        }
    }
}

#[allow(dead_code)]
fn compare_rotations(data: &[u8], a: usize, b: usize) -> Ordering {
    // Kept for reference / debugging.
    let n = data.len();
    for k in 0..n {
        let pa = (a + k) % n;
        let pb = (b + k) % n;
        match data[pa].cmp(&data[pb]) {
            Ordering::Equal => continue,
            other => return other,
        }
    }
    Ordering::Equal
}

// =========================================================================
// Multi-table Huffman
// =========================================================================
//
// MVP: always use 2 tables.  Initialize them by splitting the symbol stream
// into two halves and counting separately.  One refinement pass: re-assign
// each 50-symbol group to whichever table gives a lower cost, then rebuild.

struct HufTable {
    code_lens: Vec<u8>,
    codes: Vec<u32>,
}

fn build_huffman_tables(symbols: &[u16], alpha_size: usize) -> (Vec<HufTable>, Vec<u8>) {
    // Number of selector groups.
    let num_groups = (symbols.len() + HUFFMAN_GROUP_SIZE - 1) / HUFFMAN_GROUP_SIZE;
    debug_assert!(num_groups > 0);

    // bzip2 spec dictates table-count selection by total symbol count, but
    // we keep it simple and always use 2 (the minimum).  This is suboptimal
    // for large blocks; tracked in project_zstd_ratio_gap-style follow-up.
    let num_tables = 2usize;

    // Initial tables: split groups into halves and count symbols in each.
    let mut counts: Vec<Vec<u32>> = vec![vec![0u32; alpha_size]; num_tables];
    for (g, chunk) in symbols.chunks(HUFFMAN_GROUP_SIZE).enumerate() {
        let t = if g * 2 < num_groups { 0 } else { 1 };
        for &s in chunk {
            counts[t][s as usize] += 1;
        }
    }

    // Build tables from initial counts.
    let mut tables: Vec<HufTable> = counts
        .iter()
        .map(|c| build_huffman_from_counts(c, alpha_size))
        .collect();

    // Selector list: assign each group to the cheaper table.
    let selectors = assign_selectors(symbols, &tables);

    // One refinement pass: recount per table from the chosen selectors,
    // rebuild, re-assign.
    for _ in 0..3 {
        let mut new_counts: Vec<Vec<u32>> = vec![vec![0u32; alpha_size]; num_tables];
        for (g, chunk) in symbols.chunks(HUFFMAN_GROUP_SIZE).enumerate() {
            let t = selectors[g] as usize;
            for &s in chunk {
                new_counts[t][s as usize] += 1;
            }
        }
        // For tables that received zero symbols, fall back to the original
        // counts so we don't end up with an all-zero table.
        for (t, c) in new_counts.iter_mut().enumerate() {
            if c.iter().all(|&x| x == 0) {
                *c = counts[t].clone();
            }
        }
        tables = new_counts
            .iter()
            .map(|c| build_huffman_from_counts(c, alpha_size))
            .collect();
        let _ = assign_selectors(symbols, &tables);
    }
    let selectors = assign_selectors(symbols, &tables);

    debug_assert!(selectors.len() <= MAX_SELECTORS);
    (tables, selectors)
}

fn assign_selectors(symbols: &[u16], tables: &[HufTable]) -> Vec<u8> {
    let mut sels = Vec::with_capacity((symbols.len() + HUFFMAN_GROUP_SIZE - 1) / HUFFMAN_GROUP_SIZE);
    for chunk in symbols.chunks(HUFFMAN_GROUP_SIZE) {
        let mut best_t = 0u8;
        let mut best_cost = u64::MAX;
        for (t, table) in tables.iter().enumerate() {
            let cost: u64 = chunk
                .iter()
                .map(|&s| table.code_lens[s as usize] as u64)
                .sum();
            if cost < best_cost {
                best_cost = cost;
                best_t = t as u8;
            }
        }
        sels.push(best_t);
    }
    sels
}

/// Build a length-limited canonical Huffman code from symbol counts.  Code
/// lengths are clamped to `MAX_HUFFMAN_CODE_LEN` (17 in practice) using a
/// simple iterative weight-shifting fallback when the natural Huffman tree
/// produces a longer code.
fn build_huffman_from_counts(counts: &[u32], alpha_size: usize) -> HufTable {
    let mut code_lens = vec![0u8; alpha_size];

    // Promote a count of 0 to 1 so every symbol gets a (possibly long) code.
    // The decoder rejects code_lens of 0, so the encoder must assign a code
    // for every alphabet symbol even if it never occurs.
    let mut weights: Vec<u64> = counts.iter().map(|&c| (c.max(1) as u64) << 8).collect();

    loop {
        let mut maybe_lens = huffman_lengths_from_weights(&weights);
        let max_len = *maybe_lens.iter().max().unwrap_or(&0);
        if max_len <= MAX_HUFFMAN_CODE_LEN {
            std::mem::swap(&mut code_lens, &mut maybe_lens);
            break;
        }
        // Compress weight differences (the "old bzip2 trick"): divide all
        // weights by 2 and round up, which shrinks the dynamic range.
        for w in weights.iter_mut() {
            *w = (*w >> 1) + 1;
        }
    }

    // Generate canonical codes from lengths: shorter codes get smaller
    // bit patterns; within the same length, lower symbol indexes first.
    let mut codes = vec![0u32; alpha_size];
    let max_len = *code_lens.iter().max().unwrap_or(&0);
    if max_len == 0 {
        return HufTable { code_lens, codes };
    }
    let mut bl_count = vec![0u32; max_len as usize + 1];
    for &l in &code_lens {
        bl_count[l as usize] += 1;
    }
    let mut next_code = vec![0u32; max_len as usize + 1];
    let mut code = 0u32;
    bl_count[0] = 0;
    for bits in 1..=max_len as usize {
        code = (code + bl_count[bits - 1]) << 1;
        next_code[bits] = code;
    }
    for sym in 0..alpha_size {
        let l = code_lens[sym];
        if l > 0 {
            codes[sym] = next_code[l as usize];
            next_code[l as usize] += 1;
        }
    }
    HufTable { code_lens, codes }
}

/// Build a Huffman tree from weights and return the resulting code lengths.
/// Uses a simple priority queue.
fn huffman_lengths_from_weights(weights: &[u64]) -> Vec<u8> {
    let n = weights.len();
    if n == 0 {
        return Vec::new();
    }
    if n == 1 {
        return vec![1];
    }
    // Each tree node: (weight, parent_index, ?)
    // We use the "depth from running merges" approach.
    #[derive(Clone, Copy, Eq, PartialEq)]
    struct Node {
        weight: u64,
        parent: i32, // -1 if root
    }
    let mut nodes: Vec<Node> = (0..n)
        .map(|i| Node {
            weight: weights[i],
            parent: -1,
        })
        .collect();
    // Active set as a min-heap of (weight, index).
    use std::collections::BinaryHeap;
    use std::cmp::Reverse;
    let mut heap: BinaryHeap<Reverse<(u64, usize)>> = BinaryHeap::with_capacity(n);
    for (i, w) in weights.iter().enumerate() {
        heap.push(Reverse((*w, i)));
    }
    while heap.len() >= 2 {
        let Reverse((wa, a)) = heap.pop().unwrap();
        let Reverse((wb, b)) = heap.pop().unwrap();
        let new_idx = nodes.len();
        nodes.push(Node {
            weight: wa + wb,
            parent: -1,
        });
        nodes[a].parent = new_idx as i32;
        nodes[b].parent = new_idx as i32;
        heap.push(Reverse((wa + wb, new_idx)));
    }

    // Compute depth = code length for each leaf.
    let mut lens = vec![0u8; n];
    for leaf in 0..n {
        let mut d: u32 = 0;
        let mut cur = nodes[leaf].parent;
        while cur != -1 {
            d += 1;
            cur = nodes[cur as usize].parent;
        }
        lens[leaf] = d.min(255) as u8;
    }
    // If only one leaf had nonzero weight, our tree may give it a length of 0.
    // bzip2 requires length >= 1.
    for l in lens.iter_mut() {
        if *l == 0 {
            *l = 1;
        }
    }
    lens
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rle1_no_run() {
        assert_eq!(forward_rle1(b"abc"), b"abc".to_vec());
    }

    #[test]
    fn rle1_short_run_passthrough() {
        // 3 of the same byte stays unencoded.
        assert_eq!(forward_rle1(b"aaa"), b"aaa".to_vec());
    }

    #[test]
    fn rle1_exact_4() {
        // 4 → "aaaa\0"
        assert_eq!(forward_rle1(b"aaaa"), vec![b'a', b'a', b'a', b'a', 0]);
    }

    #[test]
    fn rle1_long_run() {
        // 100 of 'x' → "xxxx<96>"
        let input = vec![b'x'; 100];
        let out = forward_rle1(&input);
        assert_eq!(out, vec![b'x', b'x', b'x', b'x', 96]);
    }

    #[test]
    fn bwt_round_trip_via_inverse() {
        // Forward then inverse should recover the original.
        let input = b"BANANA";
        let (last, origin) = forward_bwt(input);
        let recovered = super::super::decode::test_only_inverse_bwt(&last, origin).unwrap();
        assert_eq!(recovered, input);
    }
}
