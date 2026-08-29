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
//! BWT uses SA-IS (Suffix Array - Induced Sort) on the doubled input plus a
//! sentinel for linear-time cyclic rotation sorting.  Multi-table Huffman uses
//! the bzip2 spec's 2..=6 table count with 4 selector-refinement iterations.

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
    // bzip2's decoder allocates a post-RLE1 buffer of `100000 * level - 19`
    // bytes (`nblockMAX` in bzlib_private.h).  We must split blocks so the
    // post-RLE1 size never exceeds that, otherwise the reference decoder
    // rejects the stream with "invalid data" — and crucially, *RLE1 can
    // expand* (a 4-run becomes 5 bytes), so capping the raw input at
    // `block_size` is not enough on its own.
    let nblock_max = block_size - 19;

    let mut bw = BitWriter::new();
    // File header: BZh<level>
    bw.write_bits(b'B' as u64, 8);
    bw.write_bits(b'Z' as u64, 8);
    bw.write_bits(b'h' as u64, 8);
    bw.write_bits((b'0' + level as u8) as u64, 8);

    let mut combined_crc: u32 = 0;
    let mut pos = 0usize;
    while pos < input.len() {
        // Drive RLE1 over the remaining input but stop as soon as the
        // post-RLE1 byte count would exceed `nblock_max`.  The function
        // returns how many *raw* input bytes it consumed.
        let (rle1_out, consumed) = forward_rle1_capped(&input[pos..], nblock_max);

        // CRC is computed over the *raw* input bytes (pre-RLE1) per the
        // bzip2 spec — see `BZ2_bsW` / `s->blockCRC` in bzlib's compress.c.
        let raw_block = &input[pos..pos + consumed];
        let block_crc = compute_block_crc(raw_block);
        combined_crc = combined_crc.rotate_left(1) ^ block_crc;

        encode_block_from_rle1(&mut bw, rle1_out, block_crc);
        pos += consumed;
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

fn encode_block_from_rle1(bw: &mut BitWriter, rle1_out: Vec<u8>, block_crc: u32) {
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

    // MTF state — fixed-size 256-entry stacks let LLVM avoid bounds checks and
    // keep everything in L1.  `mtf[i]` = alphabet index currently at MTF
    // position `i`; `mtf_pos[a]` = position of alphabet index `a` (inverse map
    // for O(1) lookup, replacing the linear `iter().position(...)` scan).
    let mut mtf: [u16; 256] = [0; 256];
    let mut mtf_pos: [u16; 256] = [0; 256];
    for i in 0..num_used {
        mtf[i] = i as u16;
        mtf_pos[i] = i as u16;
    }
    let alpha_size = num_used + 2; // RUNA + RUNB + (num_used - 1) MTF symbols + EOB
    let eob_symbol = (num_used + 1) as u16;

    // Output: a sequence of u16 alphabet symbols (RUNA=0, RUNB=1, MTF1..=num_used,
    // EOB=num_used+1).
    let mut symbols: Vec<u16> = Vec::with_capacity(bwt_out.len() + 1);
    let mut zero_run: u32 = 0;

    for &b in &bwt_out {
        let alpha_idx = byte_to_alpha[b as usize];
        // O(1) MTF position lookup via the inverse table.
        let pos = mtf_pos[alpha_idx as usize] as usize;

        if pos == 0 {
            zero_run += 1;
        } else {
            // Move-to-front: shift mtf[0..pos] one slot right, then update
            // mtf_pos for each shifted element.  We do the shift with
            // `copy_within` (SIMD memmove) and the position-table update in
            // a tight sweep.  `unsafe` skips bounds checks; `pos < 256` and
            // both arrays are length 256 so the indices are always valid.
            unsafe {
                let mtf_ptr = mtf.as_mut_ptr();
                let pos_ptr = mtf_pos.as_mut_ptr();
                // Equivalent to mtf.copy_within(0..pos, 1).
                std::ptr::copy(mtf_ptr, mtf_ptr.add(1), pos);
                // Sweep updates mtf_pos[mtf[i]] = i for i in 1..=pos.
                let mut i = 1usize;
                while i + 4 <= pos + 1 {
                    let a0 = *mtf_ptr.add(i);
                    let a1 = *mtf_ptr.add(i + 1);
                    let a2 = *mtf_ptr.add(i + 2);
                    let a3 = *mtf_ptr.add(i + 3);
                    *pos_ptr.add(a0 as usize) = i as u16;
                    *pos_ptr.add(a1 as usize) = (i + 1) as u16;
                    *pos_ptr.add(a2 as usize) = (i + 2) as u16;
                    *pos_ptr.add(a3 as usize) = (i + 3) as u16;
                    i += 4;
                }
                while i <= pos {
                    let a = *mtf_ptr.add(i);
                    *pos_ptr.add(a as usize) = i as u16;
                    i += 1;
                }
                *mtf_ptr = alpha_idx;
                *pos_ptr.add(alpha_idx as usize) = 0;
            }

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
        // Linear scan over at most 6 entries — branch-prediction-friendly.
        let mut pos = 0usize;
        while sel_pos[pos] != s {
            pos += 1;
        }
        // Move to front.
        let v = sel_pos[pos];
        let mut i = pos;
        while i > 0 {
            sel_pos[i] = sel_pos[i - 1];
            i -= 1;
        }
        sel_pos[0] = v;
        // Write `pos` ones followed by a zero as a single bit-field.
        // (1 << (pos+1)) - 2 is `pos` ones followed by a zero, MSB-first.
        let bits = (1u64 << (pos + 1)) - 2;
        bw.write_bits(bits, (pos + 1) as u32);
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

    // Encoded data: write each symbol with the table chosen by its 50-symbol
    // group.  Hoist the table-lookup out of the inner loop so the per-symbol
    // path is just two array accesses + write_bits.
    for (g, chunk) in symbols.chunks(HUFFMAN_GROUP_SIZE).enumerate() {
        let table = &tables[selectors[g] as usize];
        let codes = &table.codes;
        let code_lens = &table.code_lens;
        for &sym in chunk {
            let s = sym as usize;
            bw.write_bits(codes[s] as u64, code_lens[s] as u32);
        }
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
#[cfg(test)]
fn forward_rle1(input: &[u8]) -> Vec<u8> {
    let (out, _consumed) = forward_rle1_capped(input, usize::MAX);
    out
}

/// RLE1 with a hard cap on the output size in bytes.  Returns the encoded
/// output and the number of *input* bytes consumed.  Stops early if the
/// next run wouldn't fit within `max_out` — this is what lets the encoder
/// guarantee a post-RLE1 block size ≤ `block_size`, which is the bzip2
/// decoder's hard buffer limit.  Worst-case expansion: a 4-run encodes as
/// 5 bytes, so the largest output any single step adds is 5 bytes.
fn forward_rle1_capped(input: &[u8], max_out: usize) -> (Vec<u8>, usize) {
    let mut out = Vec::with_capacity(input.len().min(max_out) + 8);
    let mut i = 0usize;
    while i < input.len() {
        let b = input[i];
        let mut run = 1usize;
        while i + run < input.len() && input[i + run] == b && run < 255 {
            run += 1;
        }
        // Bytes this step will append.
        let step_out = if run >= 4 { 5 } else { run };
        if out.len() + step_out > max_out {
            break;
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
    (out, i)
}

// =========================================================================
// Forward BWT
// =========================================================================
//
// We need to sort all `n` cyclic rotations of `input`.  Two strategies:
//
//   * SA-IS over `input ++ input ++ [sentinel]` (linear-time suffix sort)
//     filtered to positions in [0, n).  Wins decisively for inputs with
//     long LCPs (text, source code, structured binaries).
//
//   * Prefix-doubling over rotations (Manber-Myers) with 2-pass LSD radix
//     sort per doubling step.  Wins for high-entropy inputs (random,
//     incompressible) where 1–2 doubling passes are enough to fully
//     disambiguate every rotation.
//
// We pick between them with a cheap heuristic: count distinct 2-grams.
// Random-like inputs have ~65k distinct 2-grams (close to n); text-like
// inputs have far fewer.  Threshold at n/2.
fn forward_bwt(input: &[u8]) -> (Vec<u8>, usize) {
    let n = input.len();
    if n == 0 {
        return (Vec::new(), 0);
    }
    if n == 1 {
        return (vec![input[0]], 0);
    }

    match bwt_via_main_sort(input) {
        Some(result) => result,
        None => bwt_via_sais(input),
    }
}

// =========================================================================
// mainSort BWT — port of C bzip2's blocksort.c
// =========================================================================
//
// Sorts cyclic rotations of `input` exactly the way `BZ2_blockSort` does:
//   1. Radix sort by first 2 bytes into 65536 small buckets (ftab).
//   2. Process the 256 big buckets in ascending-size order (runningOrder).
//   3. Sort each unsorted small bucket [ss, j != ss] with a 3-way quicksort
//      on the byte at depth `d` (mainQSort3), dropping to a shell sort
//      (mainSimpleSort + mainGtU) for ranges < 20 or depth > 14.
//   4. After big bucket ss is sorted, synthesise every [t, ss] small bucket
//      (including [ss, ss]) from predecessors in a bidirectional scan.
//   5. Store each position's rank within its big bucket in a u16 quadrant
//      (right-shifted so the largest bucket fits 16 bits — ordering is
//      preserved, so it stays a valid tiebreak) for mainGtU's deep compares.
//   6. Comparison budget (9n, one tick per 8 deep bytes). Returns None if
//      exceeded so the caller falls back to SA-IS.
//
// The block carries a 34-byte cyclic overshoot so the unrolled compares
// never need a wrap check.

const BZ_OVERSHOOT: usize = 34;
const BZ_N_RADIX: i32 = 2;
const BZ_N_QSORT: i32 = 12;
const MAIN_QSORT_SMALL_THRESH: i32 = 20;
const MAIN_QSORT_DEPTH_THRESH: i32 = BZ_N_RADIX + BZ_N_QSORT;
const MAIN_QSORT_STACK_SIZE: usize = 100;
const SHELL_INCS: [i32; 14] = [
    1, 4, 13, 40, 121, 364, 1093, 3280, 9841, 29524, 88573, 265720, 797161, 2391484,
];
/// Bit flag in ftab[] marking a small bucket as already sorted.
const SETMASK: u32 = 1 << 21;
const CLEARMASK: u32 = !SETMASK;

fn bwt_via_main_sort(input: &[u8]) -> Option<(Vec<u8>, usize)> {
    let n = input.len();

    let mut block = vec![0u8; n + BZ_OVERSHOOT];
    block[..n].copy_from_slice(input);
    for i in 0..BZ_OVERSHOOT {
        block[n + i] = input[i % n];
    }
    let mut quadrant = vec![0u16; n + BZ_OVERSHOOT];
    let mut ftab = vec![0u32; 65537];
    let mut ptr = vec![0u32; n];
    // C: budget = nblock * ((workFactor - 1) / 3), workFactor = 30.
    let mut budget: i32 = (n as i32).saturating_mul(9);

    for i in 0..n {
        let j = ((block[i] as usize) << 8) | block[i + 1] as usize;
        ftab[j] += 1;
    }
    for i in 1..65537 {
        ftab[i] += ftab[i - 1];
    }
    for i in (0..n).rev() {
        let j = ((block[i] as usize) << 8) | block[i + 1] as usize;
        ftab[j] -= 1;
        ptr[ftab[j] as usize] = i as u32;
    }

    // Big buckets in ascending size so the predecessor-copy trick handles
    // as much of the large ones as possible.
    let mut running_order: [u8; 256] = core::array::from_fn(|i| i as u8);
    let big_freq: [u32; 256] = core::array::from_fn(|b| ftab[(b + 1) << 8] - ftab[b << 8]);
    running_order.sort_by_key(|&b| big_freq[b as usize]);

    let mut big_done = [false; 256];
    let mut copy_start = [0i32; 256];
    let mut copy_end = [0i32; 256];

    for &ss in running_order.iter() {
        let ss = ss as usize;

        // Step 1: complete big bucket [ss] by sorting every small bucket
        // [ss, j], j != ss, not already synthesised.
        for j in 0..256usize {
            if j == ss {
                continue;
            }
            let sb = (ss << 8) + j;
            if ftab[sb] & SETMASK == 0 {
                let lo = (ftab[sb] & CLEARMASK) as i32;
                let hi = (ftab[sb + 1] & CLEARMASK) as i32 - 1;
                if hi > lo {
                    main_qsort3(&mut ptr, &block, &quadrant, n, lo, hi, BZ_N_RADIX, &mut budget);
                    if budget < 0 {
                        return None;
                    }
                }
            }
            ftab[sb] |= SETMASK;
        }
        debug_assert!(!big_done[ss]);

        // Step 2: scan big bucket [ss] to synthesise the sorted order of
        // every small bucket [t, ss] — including, via the moving bounds,
        // [ss, ss] itself.
        for j in 0..256usize {
            copy_start[j] = (ftab[(j << 8) + ss] & CLEARMASK) as i32;
            copy_end[j] = (ftab[(j << 8) + ss + 1] & CLEARMASK) as i32 - 1;
        }
        let mut j = (ftab[ss << 8] & CLEARMASK) as i32;
        while j < copy_start[ss] {
            let p = ptr[j as usize] as usize;
            let k = if p == 0 { n - 1 } else { p - 1 };
            let c1 = block[k] as usize;
            if !big_done[c1] {
                ptr[copy_start[c1] as usize] = k as u32;
                copy_start[c1] += 1;
            }
            j += 1;
        }
        let mut j = (ftab[(ss + 1) << 8] & CLEARMASK) as i32 - 1;
        while j > copy_end[ss] {
            let p = ptr[j as usize] as usize;
            let k = if p == 0 { n - 1 } else { p - 1 };
            let c1 = block[k] as usize;
            if !big_done[c1] {
                ptr[copy_end[c1] as usize] = k as u32;
                copy_end[c1] -= 1;
            }
            j -= 1;
        }
        debug_assert!(
            copy_start[ss] - 1 == copy_end[ss] || (copy_start[ss] == 0 && copy_end[ss] == n as i32 - 1)
        );
        for j in 0..256usize {
            ftab[(j << 8) + ss] |= SETMASK;
        }

        // Step 3: big bucket [ss] is done — record ranks in the quadrant.
        big_done[ss] = true;
        let bb_start = (ftab[ss << 8] & CLEARMASK) as usize;
        let bb_size = (ftab[(ss + 1) << 8] & CLEARMASK) as usize - bb_start;
        let mut shifts = 0u32;
        while (bb_size >> shifts) > 65534 {
            shifts += 1;
        }
        for j in (0..bb_size).rev() {
            let a2update = ptr[bb_start + j] as usize;
            let q = (j >> shifts) as u16;
            quadrant[a2update] = q;
            if a2update < BZ_OVERSHOOT {
                quadrant[a2update + n] = q;
            }
        }
    }

    let mut last = Vec::with_capacity(n);
    let mut origin = 0usize;
    for (row, &p) in ptr.iter().enumerate() {
        let p = p as usize;
        last.push(input[if p == 0 { n - 1 } else { p - 1 }]);
        if p == 0 {
            origin = row;
        }
    }
    Some((last, origin))
}

#[inline(always)]
fn mmed3(mut a: u8, mut b: u8, c: u8) -> u8 {
    if a > b {
        core::mem::swap(&mut a, &mut b);
    }
    if b > c {
        b = c;
        if a > b {
            b = a;
        }
    }
    b
}

/// 3-way quicksort of ptr[lo..=hi] on the byte at depth `d` (C: mainQSort3).
/// Every position in the range shares its first `d` bytes.
#[inline(never)]
fn main_qsort3(
    ptr: &mut [u32],
    block: &[u8],
    quadrant: &[u16],
    nblock: usize,
    lo_st: i32,
    hi_st: i32,
    d_st: i32,
    budget: &mut i32,
) {
    let mut stack = [(0i32, 0i32, 0i32); MAIN_QSORT_STACK_SIZE];
    let mut sp = 0usize;
    stack[sp] = (lo_st, hi_st, d_st);
    sp += 1;

    // SAFETY (all get_unchecked below): lo..=hi stay inside ptr (they come
    // from ftab sub-bucket bounds), and ptr[x] + d ≤ nblock - 1 + 15 which
    // is inside the 34-byte overshoot.
    while sp > 0 {
        assert!(sp < MAIN_QSORT_STACK_SIZE - 2);
        sp -= 1;
        let (lo, hi, d) = stack[sp];

        if hi - lo < MAIN_QSORT_SMALL_THRESH || d > MAIN_QSORT_DEPTH_THRESH {
            main_simple_sort(ptr, block, quadrant, nblock, lo, hi, d, budget);
            if *budget < 0 {
                return;
            }
            continue;
        }

        let at = |ptr: &[u32], i: i32| -> u8 {
            unsafe { *block.get_unchecked(*ptr.get_unchecked(i as usize) as usize + d as usize) }
        };
        let med = mmed3(at(ptr, lo), at(ptr, hi), at(ptr, (lo + hi) >> 1)) as i32;

        let mut un_lo = lo;
        let mut lt_lo = lo;
        let mut un_hi = hi;
        let mut gt_hi = hi;
        loop {
            loop {
                if un_lo > un_hi {
                    break;
                }
                let v = at(ptr, un_lo) as i32 - med;
                if v == 0 {
                    ptr.swap(un_lo as usize, lt_lo as usize);
                    lt_lo += 1;
                    un_lo += 1;
                    continue;
                }
                if v > 0 {
                    break;
                }
                un_lo += 1;
            }
            loop {
                if un_lo > un_hi {
                    break;
                }
                let v = at(ptr, un_hi) as i32 - med;
                if v == 0 {
                    ptr.swap(un_hi as usize, gt_hi as usize);
                    gt_hi -= 1;
                    un_hi -= 1;
                    continue;
                }
                if v < 0 {
                    break;
                }
                un_hi -= 1;
            }
            if un_lo > un_hi {
                break;
            }
            ptr.swap(un_lo as usize, un_hi as usize);
            un_lo += 1;
            un_hi -= 1;
        }
        debug_assert!(un_hi == un_lo - 1);

        if gt_hi < lt_lo {
            // Everything equal to the pivot: go one byte deeper.
            stack[sp] = (lo, hi, d + 1);
            sp += 1;
            continue;
        }

        let n = (lt_lo - lo).min(un_lo - lt_lo);
        vswap(ptr, lo, un_lo - n, n);
        let m = (hi - gt_hi).min(gt_hi - un_hi);
        vswap(ptr, un_lo, hi - m + 1, m);

        let n = lo + un_lo - lt_lo - 1;
        let m = hi - (gt_hi - un_hi) + 1;

        let mut next = [(lo, n, d), (m, hi, d), (n + 1, m - 1, d + 1)];
        let size = |t: &(i32, i32, i32)| t.1 - t.0;
        if size(&next[0]) < size(&next[1]) {
            next.swap(0, 1);
        }
        if size(&next[1]) < size(&next[2]) {
            next.swap(1, 2);
        }
        if size(&next[0]) < size(&next[1]) {
            next.swap(0, 1);
        }
        stack[sp] = next[0];
        stack[sp + 1] = next[1];
        stack[sp + 2] = next[2];
        sp += 3;
    }
}

#[inline(always)]
fn vswap(ptr: &mut [u32], mut p1: i32, mut p2: i32, mut n: i32) {
    while n > 0 {
        ptr.swap(p1 as usize, p2 as usize);
        p1 += 1;
        p2 += 1;
        n -= 1;
    }
}

/// Shell sort ptr[lo..=hi] with full rotation comparison (C: mainSimpleSort).
#[inline(never)]
fn main_simple_sort(
    ptr: &mut [u32],
    block: &[u8],
    quadrant: &[u16],
    nblock: usize,
    lo: i32,
    hi: i32,
    d: i32,
    budget: &mut i32,
) {
    let big_n = hi - lo + 1;
    if big_n < 2 {
        return;
    }
    let mut hp = 0usize;
    while SHELL_INCS[hp] < big_n {
        hp += 1;
    }
    let d = d as usize;
    loop {
        if hp == 0 {
            break;
        }
        hp -= 1;
        let h = SHELL_INCS[hp];
        let mut i = lo + h;
        while i <= hi {
            unsafe {
                let v = *ptr.get_unchecked(i as usize);
                let mut j = i;
                while main_gt_u(
                    block,
                    quadrant,
                    nblock,
                    *ptr.get_unchecked((j - h) as usize) as usize + d,
                    v as usize + d,
                    budget,
                ) {
                    *ptr.get_unchecked_mut(j as usize) = *ptr.get_unchecked((j - h) as usize);
                    j -= h;
                    if j <= lo + h - 1 {
                        break;
                    }
                }
                *ptr.get_unchecked_mut(j as usize) = v;
            }
            i += 1;
            if *budget < 0 {
                return;
            }
        }
    }
}

/// True if rotation starting at `i1` sorts after the one at `i2` (C: mainGtU).
/// `i1`/`i2` already include the depth offset.
#[inline(always)]
fn main_gt_u(
    block: &[u8],
    quadrant: &[u16],
    nblock: usize,
    mut i1: usize,
    mut i2: usize,
    budget: &mut i32,
) -> bool {
    debug_assert!(i1 != i2);
    unsafe {
        macro_rules! cmp1 {
            () => {
                let c1 = *block.get_unchecked(i1);
                let c2 = *block.get_unchecked(i2);
                if c1 != c2 {
                    return c1 > c2;
                }
                i1 += 1;
                i2 += 1;
            };
        }
        cmp1!(); cmp1!(); cmp1!(); cmp1!(); cmp1!(); cmp1!();
        cmp1!(); cmp1!(); cmp1!(); cmp1!(); cmp1!(); cmp1!();

        let mut k = nblock as i32 + 8;
        loop {
            macro_rules! cmpq {
                () => {
                    let c1 = *block.get_unchecked(i1);
                    let c2 = *block.get_unchecked(i2);
                    if c1 != c2 {
                        return c1 > c2;
                    }
                    let s1 = *quadrant.get_unchecked(i1);
                    let s2 = *quadrant.get_unchecked(i2);
                    if s1 != s2 {
                        return s1 > s2;
                    }
                    i1 += 1;
                    i2 += 1;
                };
            }
            cmpq!(); cmpq!(); cmpq!(); cmpq!();
            cmpq!(); cmpq!(); cmpq!(); cmpq!();
            if i1 >= nblock {
                i1 -= nblock;
            }
            if i2 >= nblock {
                i2 -= nblock;
            }
            k -= 8;
            *budget -= 1;
            if k < 0 {
                return false;
            }
        }
    }
}

fn bwt_via_sais(input: &[u8]) -> (Vec<u8>, usize) {
    let n = input.len();
    // Build T = input ++ input ++ [sentinel] over a 257-symbol alphabet:
    //   * bytes are mapped to 1..=256 so 0 is reserved for the sentinel
    //   * the trailing 0 is the unique smallest character
    // Length = 2n + 1.  The doubling is necessary for correctness: the SA of
    // just input$ gives a different ordering than cyclic rotation sort when
    // suffixes share a prefix that extends past the shorter one's sentinel.
    let total = 2 * n + 1;
    let mut t = vec![0u32; total];
    for i in 0..n {
        let b = (input[i] as u32) + 1;
        t[i] = b;
        t[i + n] = b;
    }
    // t[2*n] = 0 already (sentinel)

    let sa = sais(&t, 257);

    let mut last = Vec::with_capacity(n);
    let mut origin = 0usize;
    let mut row = 0usize;
    for &p in &sa {
        let p = p as usize;
        if p < n {
            let last_byte_index = if p == 0 { n - 1 } else { p - 1 };
            last.push(input[last_byte_index]);
            if p == 0 {
                origin = row;
            }
            row += 1;
        }
    }
    debug_assert_eq!(last.len(), n);
    (last, origin)
}

// =========================================================================
// SA-IS (Suffix Array Induced Sort) — Nong, Zhang, Chan (2009)
// =========================================================================
//
// `sais` computes the suffix array of `text` over an alphabet of size `k`.
// `text` MUST end with a unique smallest character (the sentinel).  The
// alphabet symbol values must lie in `0..k`.  Returns a vector `sa` of
// length `text.len()` where `sa[i]` is the starting index of the i-th
// smallest suffix.
//
// The algorithm:
//   1. Classify positions as L-type or S-type.
//   2. Identify LMS positions (S-type whose left neighbour is L-type).
//   3. Place LMS positions at the END of their character buckets.
//   4. Induced-sort L-types left-to-right (places L positions at the START
//      of buckets in order they're discovered).
//   5. Induced-sort S-types right-to-left (END of buckets).
//   6. Name LMS substrings; if all unique, the LMS sort is exact.  Otherwise
//      recurse on the reduced sequence to get the LMS sort, then redo steps
//      3–5.
//
// Reference: "Linear Suffix Array Construction by Almost Pure Induced-Sorting"
// by G. Nong, S. Zhang, W.H. Chan (2009).
fn sais(text: &[u32], k: usize) -> Vec<u32> {
    let n = text.len();
    let mut sa = vec![0u32; n];
    sais_impl(text, &mut sa, k);
    sa
}

/// Sentinel value used to mark "empty slot" in the SA during induced sort.
const SAIS_EMPTY: u32 = u32::MAX;

fn sais_impl(t: &[u32], sa: &mut [u32], k: usize) {
    let n = t.len();
    debug_assert!(n >= 2, "SA-IS requires at least 2 elements (incl. sentinel)");

    // ---- 1. Classify L/S types ----
    let mut t_type = vec![false; n];
    t_type[n - 1] = true;
    for i in (0..n - 1).rev() {
        t_type[i] = if t[i] < t[i + 1] {
            true
        } else if t[i] > t[i + 1] {
            false
        } else {
            t_type[i + 1]
        };
    }

    // ---- 2. Bucket sizes (count of each character in t) ----
    let mut bucket = vec![0u32; k];
    for &c in t {
        bucket[c as usize] += 1;
    }

    // Reusable scratch for bucket starts/ends (avoids repeated allocations).
    let mut bkt = vec![0u32; k];

    // ---- 3. Place LMS positions at the END of their buckets ----
    for s in sa.iter_mut() {
        *s = SAIS_EMPTY;
    }
    sais_fill_bucket_ends(&bucket, &mut bkt);
    for i in 1..n {
        if t_type[i] && !t_type[i - 1] {
            let c = t[i] as usize;
            bkt[c] -= 1;
            sa[bkt[c] as usize] = i as u32;
        }
    }

    // ---- 4. Induced sort L-types ----
    sais_induced_sort_l(t, sa, &t_type, &bucket, &mut bkt);

    // ---- 5. Induced sort S-types ----
    sais_induced_sort_s(t, sa, &t_type, &bucket, &mut bkt);

    // ---- 6. Name LMS substrings ----
    let mut name_count: u32 = 0;
    let mut prev_lms: Option<usize> = None;
    let mut name_buf = vec![SAIS_EMPTY; n];
    for i in 0..n {
        let pos = sa[i];
        if pos == SAIS_EMPTY {
            continue;
        }
        let pos = pos as usize;
        if pos == 0 || !t_type[pos] || t_type[pos - 1] {
            continue;
        }
        let is_new = match prev_lms {
            None => true,
            Some(prev) => !sais_lms_substr_equal(t, &t_type, prev, pos),
        };
        if is_new {
            name_count += 1;
        }
        name_buf[pos] = name_count - 1;
        prev_lms = Some(pos);
    }
    let mut n1 = 0usize;
    for i in 0..n {
        if name_buf[i] != SAIS_EMPTY {
            sa[n1] = name_buf[i];
            n1 += 1;
        }
    }
    drop(name_buf);

    // ---- 7. Recurse if needed ----
    let mut lms_positions: Vec<u32> = Vec::with_capacity(n1);
    for i in 1..n {
        if t_type[i] && !t_type[i - 1] {
            lms_positions.push(i as u32);
        }
    }

    if (name_count as usize) < n1 {
        let mut sub_t = vec![0u32; n1];
        sub_t.copy_from_slice(&sa[..n1]);
        let mut sub_sa = vec![0u32; n1];
        sais_impl(&sub_t, &mut sub_sa, name_count as usize);
        for i in 0..n1 {
            sa[i] = lms_positions[sub_sa[i] as usize];
        }
    } else {
        let mut tmp = vec![0u32; n1];
        for i in 0..n1 {
            tmp[sa[i] as usize] = lms_positions[i];
        }
        sa[..n1].copy_from_slice(&tmp);
    }

    // ---- 8. Final placement ----
    for i in n1..n {
        sa[i] = SAIS_EMPTY;
    }
    sais_fill_bucket_ends(&bucket, &mut bkt);
    for i in (0..n1).rev() {
        let pos = sa[i] as usize;
        sa[i] = SAIS_EMPTY;
        let c = t[pos] as usize;
        bkt[c] -= 1;
        sa[bkt[c] as usize] = pos as u32;
    }

    sais_induced_sort_l(t, sa, &t_type, &bucket, &mut bkt);
    sais_induced_sort_s(t, sa, &t_type, &bucket, &mut bkt);
}

// ---- Bucket helpers (fill existing slice, no allocation) ----

#[inline]
fn sais_fill_bucket_ends(bucket: &[u32], out: &mut [u32]) {
    let mut sum = 0u32;
    for c in 0..bucket.len() {
        sum += bucket[c];
        out[c] = sum;
    }
}

#[inline]
fn sais_fill_bucket_starts(bucket: &[u32], out: &mut [u32]) {
    let mut sum = 0u32;
    for c in 0..bucket.len() {
        out[c] = sum;
        sum += bucket[c];
    }
}

// ---- SA-IS helpers using Vec<bool> for the fallback path ----

fn sais_induced_sort_l(t: &[u32], sa: &mut [u32], t_type: &[bool], bucket: &[u32], bkt: &mut [u32]) {
    let n = t.len();
    sais_fill_bucket_starts(bucket, bkt);
    for i in 0..n {
        let p = sa[i];
        if p == SAIS_EMPTY || p == 0 {
            continue;
        }
        let j = (p - 1) as usize;
        if !t_type[j] {
            let c = t[j] as usize;
            sa[bkt[c] as usize] = j as u32;
            bkt[c] += 1;
        }
    }
}

fn sais_induced_sort_s(t: &[u32], sa: &mut [u32], t_type: &[bool], bucket: &[u32], bkt: &mut [u32]) {
    let n = t.len();
    sais_fill_bucket_ends(bucket, bkt);
    for i in (0..n).rev() {
        let p = sa[i];
        if p == SAIS_EMPTY || p == 0 {
            continue;
        }
        let j = (p - 1) as usize;
        if t_type[j] {
            let c = t[j] as usize;
            bkt[c] -= 1;
            sa[bkt[c] as usize] = j as u32;
        }
    }
}

fn sais_lms_substr_equal(t: &[u32], t_type: &[bool], a: usize, b: usize) -> bool {
    let n = t.len();
    let mut i = 0usize;
    loop {
        let pa = a + i;
        let pb = b + i;
        if pa >= n || pb >= n {
            return false;
        }
        if t[pa] != t[pb] || t_type[pa] != t_type[pb] {
            return false;
        }
        if i > 0 {
            let a_lms = t_type[pa] && !t_type[pa - 1];
            let b_lms = t_type[pb] && !t_type[pb - 1];
            if a_lms && b_lms {
                return true;
            }
            if a_lms != b_lms {
                return false;
            }
        }
        i += 1;
    }
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

    // Number of Huffman tables — bzip2 selects 2..=6 based on the symbol
    // count.  More tables → tighter fit per group (smaller bitstream) at
    // the cost of more selector overhead.  These thresholds match the
    // reference encoder.
    let num_tables: usize = if symbols.len() < 200 {
        2
    } else if symbols.len() < 600 {
        3
    } else if symbols.len() < 1200 {
        4
    } else if symbols.len() < 2400 {
        5
    } else {
        6
    };

    // Initial tables: partition groups into roughly equal "slices" of the
    // symbol stream so each table gets a contiguous range to start.  This
    // matches the bzip2 reference's "split work into nTables stripes"
    // initial assignment.
    let mut counts: Vec<Vec<u32>> = vec![vec![0u32; alpha_size]; num_tables];
    {
        let mut remaining = symbols.len();
        let mut consumed = 0usize;
        for t in 0..num_tables {
            let target = remaining / (num_tables - t);
            let mut tot = 0usize;
            let mut sym_idx = consumed;
            while sym_idx < symbols.len() && tot < target {
                tot += 1;
                counts[t][symbols[sym_idx] as usize] += 1;
                sym_idx += 1;
            }
            consumed = sym_idx;
            remaining -= tot;
        }
    }

    // Build tables from initial counts.
    let mut tables: Vec<HufTable> = counts
        .iter()
        .map(|c| build_huffman_from_counts(c, alpha_size))
        .collect();

    // Save the initial counts so empty-table iterations can fall back to
    // a non-zero distribution.
    let initial_counts = counts;

    // Up to 4 refinement passes (matching the bzip2 reference's
    // BZ_N_ITERS).  Each pass: assign each group to the cheapest table,
    // rebuild that table from the symbols actually assigned to it.  Stop
    // early if the assignment hasn't changed.
    let mut selectors: Vec<u8> = Vec::with_capacity(num_groups);
    for _iter in 0..4 {
        // Reassign every group to the cheaper table.
        let new_selectors = assign_selectors(symbols, &tables);
        let stable = new_selectors == selectors;
        selectors = new_selectors;
        if stable && _iter > 0 {
            break;
        }

        // Recount per table from the new selectors.
        let mut new_counts: Vec<Vec<u32>> = vec![vec![0u32; alpha_size]; num_tables];
        for (g, chunk) in symbols.chunks(HUFFMAN_GROUP_SIZE).enumerate() {
            let t = selectors[g] as usize;
            for &s in chunk {
                new_counts[t][s as usize] += 1;
            }
        }
        // For tables that received zero symbols, fall back to the initial
        // counts so we don't end up with an all-zero table.
        for (t, c) in new_counts.iter_mut().enumerate() {
            if c.iter().all(|&x| x == 0) {
                *c = initial_counts[t].clone();
            }
        }
        tables = new_counts
            .iter()
            .map(|c| build_huffman_from_counts(c, alpha_size))
            .collect();
    }
    // Final selector pass with the latest tables.
    selectors = assign_selectors(symbols, &tables);

    debug_assert!(selectors.len() <= MAX_SELECTORS);
    (tables, selectors)
}

fn assign_selectors(symbols: &[u16], tables: &[HufTable]) -> Vec<u8> {
    let num_groups = (symbols.len() + HUFFMAN_GROUP_SIZE - 1) / HUFFMAN_GROUP_SIZE;
    let mut sels = Vec::with_capacity(num_groups);
    let nt = tables.len();
    let alpha_size = tables[0].code_lens.len();

    // Transpose code lengths so all tables' lengths for a given symbol are
    // contiguous in memory.  Layout: `lens_t[sym * MAX_HUFFMAN_TABLES + t]`.
    // Padded slot for missing tables (t >= nt) is unused; we only sum the
    // first `nt` slots in the inner loop.
    let mut lens_t: Vec<u32> = vec![0u32; alpha_size * MAX_HUFFMAN_TABLES];
    for (t, table) in tables.iter().enumerate() {
        for (s, &l) in table.code_lens.iter().enumerate() {
            lens_t[s * MAX_HUFFMAN_TABLES + t] = l as u32;
        }
    }

    for chunk in symbols.chunks(HUFFMAN_GROUP_SIZE) {
        let mut costs = [0u32; MAX_HUFFMAN_TABLES];
        // Inner loop reads `MAX_HUFFMAN_TABLES = 6` u32s per symbol, sum into
        // costs.  LLVM auto-vectorises this into a small SIMD add.
        // Safety: `s < alpha_size` (enforced by construction) and
        // `alpha_size * MAX_HUFFMAN_TABLES == lens_t.len()`, so
        // `base + 5 < lens_t.len()`.
        for &s in chunk {
            let base = (s as usize) * MAX_HUFFMAN_TABLES;
            unsafe {
                costs[0] += *lens_t.get_unchecked(base);
                costs[1] += *lens_t.get_unchecked(base + 1);
                costs[2] += *lens_t.get_unchecked(base + 2);
                costs[3] += *lens_t.get_unchecked(base + 3);
                costs[4] += *lens_t.get_unchecked(base + 4);
                costs[5] += *lens_t.get_unchecked(base + 5);
            }
        }
        // Pick the cheapest active table.
        let mut best_t = 0u8;
        let mut best_cost = costs[0];
        for t in 1..nt {
            if costs[t] < best_cost {
                best_cost = costs[t];
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
