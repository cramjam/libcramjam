//! Block entropy back end: literals (Huffman) and sequences (FSE), following
//! C zstd's `ZSTD_compressLiterals` / `HUF_compress_internal` and
//! `ZSTD_entropyCompressSeqStore_internal` decisions.

use std::sync::LazyLock;

use super::bitc::{fse_normalize_count, fse_optimal_table_log, fse_write_ncount, BitCStream, FseCTable};
use super::cparams::Strategy;
use super::fse::{LITLEN_TABLE, MATCHLEN_TABLE, PREDEFINED_LL_WEIGHTS, PREDEFINED_ML_WEIGHTS, PREDEFINED_OF_WEIGHTS};
use super::seqstore::{highbit32, SeqDef};

// ===========================================================================
// Huffman literals
// ===========================================================================

const HUF_TABLELOG_MAX: u32 = 11;

/// `HUF_CElt` layout: code value in the top bits (`<< (64 - nb)`), `nb` in the
/// low byte.
pub struct HufCTable {
    elt: [u64; 256],
    max_symbol: usize,
    table_log: u32,
}

impl HufCTable {
    #[inline(always)]
    fn nb_bits(&self, sym: usize) -> u32 {
        (self.elt[sym] & 0xFF) as u32
    }

    /// Build a length-limited canonical Huffman table from symbol counts
    /// (`counts.len() == max_symbol + 1`, at least two non-zero).
    /// `HUF_buildCTable_wksp`: `HUF_sort` + `HUF_buildTree` +
    /// `HUF_setMaxHeight` + `HUF_buildCTableFromTree`, allocation-free.
    fn build(counts: &[u32], max_table_log: u32) -> Option<Self> {
        let max_symbol = counts.len() - 1;
        if counts.iter().filter(|&&c| c > 0).count() < 2 {
            return None;
        }
        let (nodes, non_null_rank) = huf_sort_and_build_tree(counts);
        let mut nodes = nodes;
        let max_bits = huf_set_max_height(&mut nodes, non_null_rank, max_table_log);
        if max_bits > HUF_TABLELOG_MAX {
            return None;
        }
        // HUF_buildCTableFromTree: canonical values per rank, assigned in
        // symbol order.
        let mut nb_per_rank = [0u16; HUF_TABLELOG_MAX as usize + 1];
        let mut val_per_rank = [0u16; HUF_TABLELOG_MAX as usize + 1];
        let mut nb_bits = [0u8; 256];
        for n in 0..=non_null_rank {
            nb_per_rank[nodes[n].nb_bits as usize] += 1;
            nb_bits[nodes[n].byte as usize] = nodes[n].nb_bits;
        }
        let mut min = 0u16;
        for n in (1..=max_bits as usize).rev() {
            val_per_rank[n] = min;
            min += nb_per_rank[n];
            min >>= 1;
        }
        let mut elt = [0u64; 256];
        for sym in 0..=max_symbol {
            let nb = nb_bits[sym] as u32;
            if nb > 0 {
                let v = val_per_rank[nb as usize];
                val_per_rank[nb as usize] += 1;
                elt[sym] = ((v as u64) << (64 - nb)) | nb as u64;
            }
        }
        Some(HufCTable { elt, max_symbol, table_log: max_bits })
    }


    /// `HUF_writeCTable`: weights for symbols `0..max_symbol` (the last one
    /// is implicit), FSE-compressed when that is smaller, else 4-bit direct.
    fn write_table(&self, out: &mut Vec<u8>) -> Option<usize> {
        let n = self.max_symbol; // number of weights emitted
        let mut weights = vec![0u8; n];
        for (s, w) in weights.iter_mut().enumerate() {
            let nb = self.nb_bits(s);
            *w = if nb == 0 { 0 } else { (self.table_log + 1 - nb) as u8 };
        }
        if let Some(fse) = super::huf::encode_weights_fse(&weights) {
            if fse.len() > 1 && fse.len() < n / 2 {
                out.push(fse.len() as u8);
                out.extend_from_slice(&fse);
                return Some(1 + fse.len());
            }
        }
        if n > 128 {
            return None;
        }
        out.push((127 + n) as u8);
        let mut i = 0;
        while i < n {
            let hi = weights[i];
            let lo = if i + 1 < n { weights[i + 1] } else { 0 };
            out.push((hi << 4) | lo);
            i += 2;
        }
        Some(1 + (n + 1) / 2)
    }
}

#[derive(Clone, Copy, Default)]
struct HufNode {
    count: u32,
    parent: u16,
    byte: u8,
    nb_bits: u8,
}

const HUF_STARTNODE: usize = 256;
const RANK_POSITION_TABLE_SIZE: usize = 192;
const RANK_POSITION_LOG_BUCKETS_BEGIN: u32 = 158;
const RANK_POSITION_DISTINCT_COUNT_CUTOFF: u32 = 166;

#[inline]
fn huf_get_index(count: u32) -> usize {
    if count < RANK_POSITION_DISTINCT_COUNT_CUTOFF {
        count as usize
    } else {
        (31 - count.leading_zeros() + RANK_POSITION_LOG_BUCKETS_BEGIN) as usize
    }
}

/// `HUF_simpleQuickSort` (descending by count, rightmost pivot, insertion
/// sort below 8) — the exact order matters for tie-breaking parity.
fn huf_sort_bucket(a: &mut [HufNode]) {
    if a.len() < 8 {
        for i in 1..a.len() {
            let key = a[i];
            let mut j = i;
            while j > 0 && a[j - 1].count < key.count {
                a[j] = a[j - 1];
                j -= 1;
            }
            a[j] = key;
        }
        return;
    }
    let high = a.len() - 1;
    let pivot = a[high].count;
    let mut i = 0usize;
    for j in 0..high {
        if a[j].count > pivot {
            a.swap(i, j);
            i += 1;
        }
    }
    a.swap(i, high);
    let (lo, hi) = a.split_at_mut(i);
    huf_sort_bucket(lo);
    huf_sort_bucket(&mut hi[1..]);
}

/// `HUF_sort` + `HUF_buildTree`. Returns the node table (index 0 is C's
/// `huffNode0[0]` barrier; symbols start at 1) and `nonNullRank`, both
/// relative to the symbol base as in C.
fn huf_sort_and_build_tree(counts: &[u32]) -> (HufNodes, usize) {
    let n_syms = counts.len();
    let mut nodes = HufNodes([HufNode::default(); 2 * 256 + 2]);
    // HUF_sort: bucket by rank, higher counts first.
    let mut base = [0u32; RANK_POSITION_TABLE_SIZE];
    let mut curr = [0u32; RANK_POSITION_TABLE_SIZE];
    for &c in counts {
        base[huf_get_index(c)] += 1;
    }
    for n in (1..RANK_POSITION_TABLE_SIZE).rev() {
        base[n - 1] += base[n];
        curr[n - 1] = base[n - 1];
    }
    for (sym, &c) in counts.iter().enumerate() {
        let r = huf_get_index(c) + 1;
        let pos = curr[r] as usize;
        curr[r] += 1;
        nodes[pos] = HufNode { count: c, parent: 0, byte: sym as u8, nb_bits: 0 };
    }
    for n in RANK_POSITION_DISTINCT_COUNT_CUTOFF as usize..RANK_POSITION_TABLE_SIZE - 1 {
        let (b, e) = (base[n] as usize, curr[n] as usize);
        if e - b > 1 {
            huf_sort_bucket(&mut nodes.0[b + 1..e + 1]);
        }
    }
    // HUF_buildTree.
    let mut non_null_rank = n_syms - 1;
    while nodes[non_null_rank].count == 0 {
        non_null_rank -= 1;
    }
    let mut node_nb = HUF_STARTNODE;
    let mut low_s = non_null_rank as isize;
    let node_root = node_nb + low_s as usize - 1;
    let mut low_n = node_nb;
    nodes[node_nb].count = nodes[low_s as usize].count + nodes[low_s as usize - 1].count;
    nodes[low_s as usize].parent = node_nb as u16;
    nodes[low_s as usize - 1].parent = node_nb as u16;
    node_nb += 1;
    low_s -= 2;
    for n in node_nb..=node_root {
        nodes[n].count = 1 << 30;
    }
    nodes.0[0].count = 1 << 31; // huffNode0[0]: strong barrier
    while node_nb <= node_root {
        let n1 = if nodes.get(low_s).count < nodes[low_n].count {
            low_s -= 1;
            (low_s + 1) as usize
        } else {
            low_n += 1;
            low_n - 1
        };
        let n2 = if nodes.get(low_s).count < nodes[low_n].count {
            low_s -= 1;
            (low_s + 1) as usize
        } else {
            low_n += 1;
            low_n - 1
        };
        nodes[node_nb].count = nodes[n1].count + nodes[n2].count;
        nodes[n1].parent = node_nb as u16;
        nodes[n2].parent = node_nb as u16;
        node_nb += 1;
    }
    nodes[node_root].nb_bits = 0;
    for n in (HUF_STARTNODE..node_root).rev() {
        nodes[n].nb_bits = nodes[nodes[n].parent as usize].nb_bits + 1;
    }
    for n in 0..=non_null_rank {
        nodes[n].nb_bits = nodes[nodes[n].parent as usize].nb_bits + 1;
    }
    (nodes, non_null_rank)
}

/// Node table with C's `huffNode = huffNode0 + 1` indexing.
struct HufNodes([HufNode; 2 * 256 + 2]);
impl HufNodes {
    #[inline(always)]
    fn get(&self, i: isize) -> &HufNode {
        &self.0[(i + 1) as usize]
    }
}
impl std::ops::Index<usize> for HufNodes {
    type Output = HufNode;
    #[inline(always)]
    fn index(&self, i: usize) -> &HufNode {
        &self.0[i + 1]
    }
}
impl std::ops::IndexMut<usize> for HufNodes {
    #[inline(always)]
    fn index_mut(&mut self, i: usize) -> &mut HufNode {
        &mut self.0[i + 1]
    }
}

/// `HUF_setMaxHeight`: cap code lengths at `target` and repay the cost.
fn huf_set_max_height(nodes: &mut HufNodes, last_non_null: usize, target: u32) -> u32 {
    let largest = nodes[last_non_null].nb_bits as u32;
    if largest <= target {
        return largest;
    }
    let base_cost: i32 = 1 << (largest - target);
    let mut n = last_non_null as isize;
    let mut total_cost: i32 = 0;
    while nodes[n as usize].nb_bits as u32 > target {
        total_cost += base_cost - (1 << (largest - nodes[n as usize].nb_bits as u32));
        nodes[n as usize].nb_bits = target as u8;
        n -= 1;
    }
    while nodes[n as usize].nb_bits as u32 == target {
        n -= 1;
    }
    total_cost >>= largest - target;
    const NO_SYMBOL: u32 = 0xF0F0_F0F0;
    let mut rank_last = [NO_SYMBOL; HUF_TABLELOG_MAX as usize + 2];
    {
        let mut current_nb_bits = target;
        let mut pos = n;
        while pos >= 0 {
            let nb = nodes[pos as usize].nb_bits as u32;
            if nb < current_nb_bits {
                current_nb_bits = nb;
                rank_last[(target - current_nb_bits) as usize] = pos as u32;
            }
            pos -= 1;
        }
    }
    while total_cost > 0 {
        let mut nb_to_decrease = (31 - (total_cost as u32).leading_zeros()) as usize + 1;
        while nb_to_decrease > 1 {
            let high_pos = rank_last[nb_to_decrease];
            let low_pos = rank_last[nb_to_decrease - 1];
            if high_pos == NO_SYMBOL {
                nb_to_decrease -= 1;
                continue;
            }
            if low_pos == NO_SYMBOL {
                break;
            }
            let high_total = nodes[high_pos as usize].count;
            let low_total = 2 * nodes[low_pos as usize].count;
            if high_total <= low_total {
                break;
            }
            nb_to_decrease -= 1;
        }
        while nb_to_decrease <= HUF_TABLELOG_MAX as usize && rank_last[nb_to_decrease] == NO_SYMBOL {
            nb_to_decrease += 1;
        }
        total_cost -= 1 << (nb_to_decrease - 1);
        nodes[rank_last[nb_to_decrease] as usize].nb_bits += 1;
        if rank_last[nb_to_decrease - 1] == NO_SYMBOL {
            rank_last[nb_to_decrease - 1] = rank_last[nb_to_decrease];
        }
        if rank_last[nb_to_decrease] == 0 {
            rank_last[nb_to_decrease] = NO_SYMBOL;
        } else {
            rank_last[nb_to_decrease] -= 1;
            if nodes[rank_last[nb_to_decrease] as usize].nb_bits as u32 != target - nb_to_decrease as u32 {
                rank_last[nb_to_decrease] = NO_SYMBOL;
            }
        }
    }
    while total_cost < 0 {
        if rank_last[1] == NO_SYMBOL {
            while nodes[n as usize].nb_bits as u32 == target {
                n -= 1;
            }
            nodes[(n + 1) as usize].nb_bits -= 1;
            rank_last[1] = (n + 1) as u32;
            total_cost += 1;
            continue;
        }
        nodes[rank_last[1] as usize + 1].nb_bits -= 1;
        rank_last[1] += 1;
        total_cost += 1;
    }
    target
}

/// `HUF_CStream_t`: two containers filled from the top. Output goes through
/// a raw pointer into the pre-reserved `Vec` so the hot loop never touches
/// the `Vec` header.
struct HufStream {
    dst: *mut u8,
    pos: usize,
    container: [u64; 2],
    bit_pos: [u32; 2],
}

impl HufStream {
    #[inline(always)]
    fn add(&mut self, elt: u64, idx: usize) {
        let nb = (elt & 0xFF) as u32;
        self.container[idx] >>= nb;
        self.container[idx] |= elt & !0xFF;
        self.bit_pos[idx] += nb;
    }
    #[inline(always)]
    fn zero_index1(&mut self) {
        self.container[1] = 0;
        self.bit_pos[1] = 0;
    }
    #[inline(always)]
    fn merge_index1(&mut self) {
        self.container[0] >>= self.bit_pos[1];
        self.container[0] |= self.container[1];
        self.bit_pos[0] += self.bit_pos[1];
    }
    /// The valid bits are the top `bit_pos` bits (oldest lowest); write them
    /// out LE and keep the up-to-7 leftover bits where they are — the next
    /// `add` shifts them down under the new symbol.
    #[inline(always)]
    fn flush(&mut self) {
        let nb_bits = self.bit_pos[0] & 0xFF;
        let nb_bytes = (nb_bits >> 3) as usize;
        // `nb_bits == 0` only happens on an empty join step; the two-step
        // shift avoids shifting by 64.
        let data = (self.container[0] >> 1) >> (63 - nb_bits);
        unsafe {
            core::ptr::write_unaligned(self.dst.add(self.pos) as *mut u64, data.to_le());
        }
        self.pos += nb_bytes;
        self.bit_pos[0] &= 7;
    }
}

/// `HUF_compress1X_usingCTable_internal_body_loop` with C's per-tableLog
/// unroll factor `K` (a const so the inner loops unroll fully).
#[inline(never)]
fn huf_compress_1x_loop<const K: usize>(bc: &mut HufStream, src: &[u8], elt: &[u64; 256]) {
    let mut n = src.len();
    let p = src.as_ptr();
    let rem = n % K;
    for _ in 0..rem {
        n -= 1;
        bc.add(elt[src[n] as usize], 0);
    }
    bc.flush();
    if n % (2 * K) != 0 {
        for u in 1..=K {
            bc.add(elt[src[n - u] as usize], 0);
        }
        bc.flush();
        n -= K;
    }
    while n > 0 {
        unsafe {
            for u in 1..=K {
                bc.add(*elt.get_unchecked(*p.add(n - u) as usize), 0);
            }
            bc.flush();
            bc.zero_index1();
            for u in 1..=K {
                bc.add(*elt.get_unchecked(*p.add(n - K - u) as usize), 1);
            }
        }
        bc.merge_index1();
        bc.flush();
        n -= 2 * K;
    }
}

/// `HUF_compress1X_usingCTable_internal_body`: returns bytes written.
fn huf_compress_1x(out: &mut Vec<u8>, src: &[u8], ct: &HufCTable) -> usize {
    let n = src.len();
    out.reserve((n * ct.table_log as usize >> 3) + 32);
    let start = out.len();
    let mut bc = HufStream { dst: unsafe { out.as_mut_ptr().add(start) }, pos: 0, container: [0; 2], bit_pos: [0; 2] };
    match ct.table_log {
        11 | 10 => huf_compress_1x_loop::<5>(&mut bc, src, &ct.elt),
        9 => huf_compress_1x_loop::<6>(&mut bc, src, &ct.elt),
        8 => huf_compress_1x_loop::<7>(&mut bc, src, &ct.elt),
        7 => huf_compress_1x_loop::<8>(&mut bc, src, &ct.elt),
        _ => huf_compress_1x_loop::<9>(&mut bc, src, &ct.elt),
    }
    // end mark: a single 1 bit
    bc.add((1u64 << 63) | 1, 0);
    bc.flush();
    let total = bc.pos + (bc.bit_pos[0] > 0) as usize;
    debug_assert!(start + total <= out.capacity());
    unsafe { out.set_len(start + total) };
    total
}


/// `HUF_compressCTable_internal`: 1 or 4 streams; `None` when not
/// compressible.
fn huf_compress_ctable(out: &mut Vec<u8>, src: &[u8], ct: &HufCTable, single_stream: bool) -> Option<usize> {
    let start = out.len();
    if single_stream {
        let c = huf_compress_1x(out, src, ct);
        if c == 0 || c >= src.len() - 1 {
            out.truncate(start);
            return None;
        }
        return Some(c);
    }
    if src.len() < 12 {
        return None;
    }
    let seg = (src.len() + 3) / 4;
    out.extend_from_slice(&[0u8; 6]);
    let mut sizes = [0usize; 3];
    let mut ip = 0usize;
    for (i, size) in sizes.iter_mut().enumerate() {
        let c = huf_compress_1x(out, &src[ip..ip + seg], ct);
        if c == 0 || c > 65535 {
            out.truncate(start);
            return None;
        }
        *size = c;
        let _ = i;
        ip += seg;
    }
    let c = huf_compress_1x(out, &src[ip..], ct);
    if c == 0 || c > 65535 {
        out.truncate(start);
        return None;
    }
    for (i, &s) in sizes.iter().enumerate() {
        out[start + 2 * i] = s as u8;
        out[start + 2 * i + 1] = (s >> 8) as u8;
    }
    let total = out.len() - start;
    if total >= src.len() - 1 {
        out.truncate(start);
        return None;
    }
    Some(total)
}

fn histogram(src: &[u8]) -> ([u32; 256], usize) {
    let mut c = [[0u32; 256]; 4];
    let mut chunks = src.chunks_exact(4);
    for ch in &mut chunks {
        c[0][ch[0] as usize] += 1;
        c[1][ch[1] as usize] += 1;
        c[2][ch[2] as usize] += 1;
        c[3][ch[3] as usize] += 1;
    }
    for &b in chunks.remainder() {
        c[0][b as usize] += 1;
    }
    let mut counts = [0u32; 256];
    let mut max_sym = 0;
    for s in 0..256 {
        counts[s] = c[0][s] + c[1][s] + c[2][s] + c[3][s];
        if counts[s] != 0 {
            max_sym = s;
        }
    }
    (counts, max_sym)
}

fn min_literals_to_compress(strategy: Strategy) -> usize {
    let shift = (9 - strategy as i32).min(3);
    8usize << shift
}

pub fn min_gain(src_size: usize) -> usize {
    (src_size >> 6) + 2
}

fn write_raw_literals(out: &mut Vec<u8>, lits: &[u8]) {
    let n = lits.len();
    if n < 32 {
        out.push((n << 3) as u8);
    } else if n < 4096 {
        out.push((1 << 2) | ((n as u8 & 0x0F) << 4));
        out.push((n >> 4) as u8);
    } else {
        out.push((3 << 2) | ((n as u8 & 0x0F) << 4));
        out.push((n >> 4) as u8);
        out.push((n >> 12) as u8);
    }
    out.extend_from_slice(lits);
}

fn write_rle_literals(out: &mut Vec<u8>, lits: &[u8]) {
    let n = lits.len();
    if n < 32 {
        out.push(1 | (n << 3) as u8);
    } else if n < 4096 {
        out.push(1 | (1 << 2) | ((n as u8 & 0x0F) << 4));
        out.push((n >> 4) as u8);
    } else {
        out.push(1 | (3 << 2) | ((n as u8 & 0x0F) << 4));
        out.push((n >> 4) as u8);
        out.push((n >> 12) as u8);
    }
    out.push(lits[0]);
}

/// `ZSTD_compressLiterals`: append the literals section for `lits`.
pub fn compress_literals(out: &mut Vec<u8>, lits: &[u8], strategy: Strategy, suspect_uncompressible: bool) {
    let n = lits.len();
    if n < min_literals_to_compress(strategy) {
        write_raw_literals(out, lits);
        return;
    }
    let lh_size = 3 + (n >= 1024) as usize + (n >= 16 * 1024) as usize;
    let single_stream = n < 256;
    let start = out.len();

    // HUF_compress_internal
    if suspect_uncompressible && n >= 2 * 128 {
        // Sample 128 bytes at each end.
        let mut largest = 0usize;
        for sample in [&lits[..128], &lits[n - 128..]] {
            let (c, _) = histogram(sample);
            largest += *c.iter().max().unwrap() as usize;
        }
        if largest <= (256 >> 7) + 4 {
            write_raw_literals(out, lits);
            return;
        }
    }
    let (counts, max_sym) = histogram(lits);
    let largest = *counts.iter().max().unwrap() as usize;
    if largest == n {
        write_rle_literals(out, lits);
        return;
    }
    if largest <= (n >> 7) + 4 {
        write_raw_literals(out, lits);
        return;
    }
    let huff_log = fse_optimal_table_log(HUF_TABLELOG_MAX, n, max_sym as u32, 1).min(HUF_TABLELOG_MAX);
    let ct = match HufCTable::build(&counts[..=max_sym], huff_log) {
        Some(t) => t,
        None => {
            write_raw_literals(out, lits);
            return;
        }
    };
    // Header placeholder, then table + streams.
    out.resize(start + lh_size, 0);
    let h_size = match ct.write_table(out) {
        Some(h) => h,
        None => {
            out.truncate(start);
            write_raw_literals(out, lits);
            return;
        }
    };
    if h_size + 12 >= n {
        out.truncate(start);
        write_raw_literals(out, lits);
        return;
    }
    let c_size = match huf_compress_ctable(out, lits, &ct, single_stream) {
        Some(c) => h_size + c,
        None => {
            out.truncate(start);
            write_raw_literals(out, lits);
            return;
        }
    };
    if c_size >= n - min_gain(n) {
        out.truncate(start);
        write_raw_literals(out, lits);
        return;
    }
    // Header: lit_type 2 (compressed), size_format by lh_size.
    let h_type = 2u32;
    match lh_size {
        3 => {
            let lhc = h_type | ((!single_stream as u32) << 2) | ((n as u32) << 4) | ((c_size as u32) << 14);
            out[start] = lhc as u8;
            out[start + 1] = (lhc >> 8) as u8;
            out[start + 2] = (lhc >> 16) as u8;
        }
        4 => {
            let lhc = h_type | (2 << 2) | ((n as u32) << 4) | ((c_size as u32) << 18);
            out[start..start + 4].copy_from_slice(&lhc.to_le_bytes());
        }
        _ => {
            let lhc = h_type | (3 << 2) | ((n as u32) << 4) | ((c_size as u32) << 22);
            out[start..start + 4].copy_from_slice(&lhc.to_le_bytes());
            out[start + 4] = (c_size >> 10) as u8;
        }
    }
}

// ===========================================================================
// Sequences
// ===========================================================================

const MAX_LL: usize = 35;
const MAX_ML: usize = 52;
const MAX_OFF: usize = 31;
const DEFAULT_MAX_OFF: usize = 28;
const LL_FSELOG: u32 = 9;
const ML_FSELOG: u32 = 9;
const OFF_FSELOG: u32 = 8;

static PREDEF_LL: LazyLock<FseCTable> = LazyLock::new(|| FseCTable::build(&PREDEFINED_LL_WEIGHTS, 6));
static PREDEF_OF: LazyLock<FseCTable> = LazyLock::new(|| FseCTable::build(&PREDEFINED_OF_WEIGHTS, 5));
static PREDEF_ML: LazyLock<FseCTable> = LazyLock::new(|| FseCTable::build(&PREDEFINED_ML_WEIGHTS, 6));

const LL_CODE: [u8; 64] = [
    0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 16, 17, 17, 18, 18, 19, 19, 20, 20, 20, 20, 21, 21, 21, 21, 22, 22, 22,
    22, 22, 22, 22, 22, 23, 23, 23, 23, 23, 23, 23, 23, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24,
];
const ML_CODE: [u8; 128] = [
    0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 32, 33,
    33, 34, 34, 35, 35, 36, 36, 36, 36, 37, 37, 37, 37, 38, 38, 38, 38, 38, 38, 38, 38, 39, 39, 39, 39, 39, 39, 39, 39, 40, 40, 40,
    40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 41, 41, 41, 41, 41, 41, 41, 41, 41, 41, 41, 41, 41, 41, 41, 41, 42, 42, 42,
    42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42,
];

#[inline(always)]
fn ll_code(lit_len: u32) -> u8 {
    if lit_len > 63 {
        (highbit32(lit_len) + 19) as u8
    } else {
        LL_CODE[lit_len as usize]
    }
}
#[inline(always)]
fn ml_code(ml_base: u32) -> u8 {
    if ml_base > 127 {
        (highbit32(ml_base) + 36) as u8
    } else {
        ML_CODE[ml_base as usize]
    }
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum SetType {
    Basic = 0,
    Rle = 1,
    Compressed = 2,
}

static K_INVERSE_PROBABILITY_LOG256: [u32; 256] = [
    0, 2048, 1792, 1642, 1536, 1453, 1386, 1329, 1280, 1236, 1197, 1162, 1130, 1100, 1073, 1047, 1024, 1001, 980, 960, 941,
    923, 906, 889, 874, 859, 844, 830, 817, 804, 791, 779, 768, 756, 745, 734, 724, 714, 704, 694, 685, 676, 667, 658, 650,
    642, 633, 626, 618, 610, 603, 595, 588, 581, 574, 567, 561, 554, 548, 542, 535, 529, 523, 517, 512, 506, 500, 495, 489,
    484, 478, 473, 468, 463, 458, 453, 448, 443, 438, 434, 429, 424, 420, 415, 411, 407, 402, 398, 394, 390, 386, 382, 377,
    373, 370, 366, 362, 358, 354, 350, 347, 343, 339, 336, 332, 329, 325, 322, 318, 315, 311, 308, 305, 302, 298, 295, 292,
    289, 286, 282, 279, 276, 273, 270, 267, 264, 261, 258, 256, 253, 250, 247, 244, 241, 239, 236, 233, 230, 228, 225, 222,
    220, 217, 215, 212, 209, 207, 204, 202, 199, 197, 194, 192, 190, 187, 185, 182, 180, 178, 175, 173, 171, 168, 166, 164,
    162, 159, 157, 155, 153, 151, 149, 146, 144, 142, 140, 138, 136, 134, 132, 130, 128, 126, 123, 121, 119, 117, 115, 114,
    112, 110, 108, 106, 104, 102, 100, 98, 96, 94, 93, 91, 89, 87, 85, 83, 82, 80, 78, 76, 74, 73, 71, 69, 67, 66, 64, 62,
    61, 59, 57, 55, 54, 52, 50, 49, 47, 46, 44, 42, 41, 39, 37, 36, 34, 33, 31, 30, 28, 26, 25, 23, 22, 20, 19, 17, 16, 14,
    13, 11, 10, 8, 7, 5, 4, 2, 1,
];

fn entropy_cost(count: &[u32], total: usize) -> usize {
    let mut cost: u64 = 0;
    for &c in count {
        let mut norm = (256 * c as usize) / total;
        if c != 0 && norm == 0 {
            norm = 1;
        }
        cost += c as u64 * K_INVERSE_PROBABILITY_LOG256[norm] as u64;
    }
    (cost >> 8) as usize
}

fn cross_entropy_cost(norm: &[i16], accuracy_log: u32, count: &[u32]) -> usize {
    let shift = 8 - accuracy_log;
    let mut cost: u64 = 0;
    for (s, &c) in count.iter().enumerate() {
        let norm_acc = if norm[s] != -1 { norm[s] as u32 } else { 1 };
        let norm256 = norm_acc << shift;
        cost += c as u64 * K_INVERSE_PROBABILITY_LOG256[norm256 as usize] as u64;
    }
    (cost >> 8) as usize
}

fn ncount_cost(count: &[u32], nb_seq: usize, fse_log: u32) -> Option<usize> {
    let max = count.len() as u32 - 1;
    let table_log = fse_optimal_table_log(fse_log, nb_seq, max, 2);
    let norm = fse_normalize_count(count, nb_seq, table_log, nb_seq >= 2048)?;
    let mut tmp = Vec::new();
    Some(fse_write_ncount(&mut tmp, &norm, table_log))
}

/// `ZSTD_selectEncodingType` without repeat tables.
fn select_encoding_type(
    count: &[u32],
    most_frequent: usize,
    nb_seq: usize,
    fse_log: u32,
    default_norm: &[i16],
    default_norm_log: u32,
    default_allowed: bool,
    strategy: Strategy,
) -> SetType {
    if most_frequent == nb_seq {
        if default_allowed && nb_seq <= 2 {
            return SetType::Basic;
        }
        return SetType::Rle;
    }
    if strategy < Strategy::Lazy {
        if default_allowed {
            let mult = 10 - strategy as usize;
            let dynamic_fse_nb_seq_min = ((1usize << default_norm_log) * mult) >> 3;
            if nb_seq < dynamic_fse_nb_seq_min || most_frequent < (nb_seq >> (default_norm_log - 1)) {
                return SetType::Basic;
            }
        }
    } else {
        let basic_cost = if default_allowed {
            let max = count.len();
            cross_entropy_cost(&default_norm[..max.min(default_norm.len())], default_norm_log, &count[..max.min(default_norm.len())])
        } else {
            usize::MAX
        };
        let compressed_cost = match ncount_cost(count, nb_seq, fse_log) {
            Some(n) => (n << 3) + entropy_cost(count, nb_seq),
            None => usize::MAX,
        };
        if basic_cost <= compressed_cost {
            return SetType::Basic;
        }
    }
    SetType::Compressed
}

enum Coder {
    Predef(&'static FseCTable),
    Own(FseCTable),
}

impl Coder {
    fn table(&self) -> &FseCTable {
        match self {
            Coder::Predef(t) => t,
            Coder::Own(t) => t,
        }
    }
}

/// `ZSTD_buildCTable`: choose the table, write its description, return the
/// coder.
fn build_ctable(
    out: &mut Vec<u8>,
    set: SetType,
    count: &mut [u32],
    codes: &[u8],
    nb_seq: usize,
    fse_log: u32,
    predef: &'static FseCTable,
) -> Coder {
    match set {
        SetType::Rle => {
            out.push(codes[0]);
            Coder::Own(FseCTable::rle(codes[0]))
        }
        SetType::Basic => Coder::Predef(predef),
        SetType::Compressed => {
            let max = count.len() as u32 - 1;
            let table_log = fse_optimal_table_log(fse_log, nb_seq, max, 2);
            let mut nb_seq_1 = nb_seq;
            let last = codes[nb_seq - 1] as usize;
            if count[last] > 1 {
                count[last] -= 1;
                nb_seq_1 -= 1;
            }
            let norm = fse_normalize_count(count, nb_seq_1, table_log, nb_seq_1 >= 2048).expect("normalize");
            fse_write_ncount(out, &norm, table_log);
            Coder::Own(FseCTable::build(&norm, table_log))
        }
    }
}

fn count_codes(codes: &[u8], max: usize) -> (Vec<u32>, usize, usize) {
    let mut count = vec![0u32; max + 1];
    for &c in codes {
        count[c as usize] += 1;
    }
    let most = *count.iter().max().unwrap() as usize;
    let last = count.iter().rposition(|&c| c != 0).unwrap_or(0);
    count.truncate(last + 1);
    (count, most, last)
}

/// `ZSTD_encodeSequences_body`.
fn encode_sequences(
    out: &mut Vec<u8>,
    seqs: &[SeqDef],
    ll_codes: &[u8],
    ml_codes: &[u8],
    of_codes: &[u8],
    ct_ll: &FseCTable,
    ct_of: &FseCTable,
    ct_ml: &FseCTable,
) {
    let nb_seq = seqs.len();
    let mut bits = BitCStream::new(out, nb_seq * 12 + 32);
    let last = nb_seq - 1;
    let mut st_ml = ct_ml.init_state(ml_codes[last]);
    let mut st_of = ct_of.init_state(of_codes[last]);
    let mut st_ll = ct_ll.init_state(ll_codes[last]);
    bits.add_bits(seqs[last].lit_len, LITLEN_TABLE[ll_codes[last] as usize].1 as u32);
    bits.add_bits(seqs[last].ml_base, MATCHLEN_TABLE[ml_codes[last] as usize].1 as u32);
    bits.add_bits(seqs[last].off_base, of_codes[last] as u32);
    bits.flush();

    let sp = seqs.as_ptr();
    for n in (0..last).rev() {
        // SAFETY: n < nb_seq and the code tables are nb_seq long.
        unsafe {
            let ll_c = *ll_codes.get_unchecked(n);
            let of_c = *of_codes.get_unchecked(n);
            let ml_c = *ml_codes.get_unchecked(n);
            let ll_bits = LITLEN_TABLE.get_unchecked(ll_c as usize).1 as u32;
            let of_bits = of_c as u32;
            let ml_bits = MATCHLEN_TABLE.get_unchecked(ml_c as usize).1 as u32;
            let s = *sp.add(n);
            ct_of.encode(&mut st_of, of_c, &mut bits);
            ct_ml.encode(&mut st_ml, ml_c, &mut bits);
            ct_ll.encode(&mut st_ll, ll_c, &mut bits);
            if of_bits + ml_bits + ll_bits >= 64 - 7 - (LL_FSELOG + ML_FSELOG + OFF_FSELOG) {
                bits.flush();
            }
            bits.add_bits(s.lit_len, ll_bits);
            bits.add_bits(s.ml_base, ml_bits);
            if of_bits + ml_bits + ll_bits > 56 {
                bits.flush();
            }
            bits.add_bits(s.off_base, of_bits);
            bits.flush();
        }
    }
    ct_ml.flush_state(&st_ml, &mut bits);
    ct_of.flush_state(&st_of, &mut bits);
    ct_ll.flush_state(&st_ll, &mut bits);
    bits.close();
}

/// Sequences section (`nbSeq` header, modes, tables, bitstream). `None`
/// means the caller should fall back to a raw block.
pub fn compress_sequences(out: &mut Vec<u8>, seqs: &[SeqDef], strategy: Strategy) -> Option<()> {
    let nb_seq = seqs.len();
    if nb_seq < 128 {
        out.push(nb_seq as u8);
    } else if nb_seq < 0x7F00 {
        out.push(((nb_seq >> 8) | 0x80) as u8);
        out.push(nb_seq as u8);
    } else {
        out.push(0xFF);
        let v = nb_seq - 0x7F00;
        out.push(v as u8);
        out.push((v >> 8) as u8);
    }
    if nb_seq == 0 {
        return Some(());
    }

    // ZSTD_seqToCodes
    let mut ll_codes = vec![0u8; nb_seq];
    let mut ml_codes = vec![0u8; nb_seq];
    let mut of_codes = vec![0u8; nb_seq];
    for (i, s) in seqs.iter().enumerate() {
        ll_codes[i] = ll_code(s.lit_len);
        ml_codes[i] = ml_code(s.ml_base);
        of_codes[i] = highbit32(s.off_base) as u8;
    }

    let seq_head = out.len();
    out.push(0);

    let (mut ll_count, ll_most, _) = count_codes(&ll_codes, MAX_LL);
    let ll_type = select_encoding_type(&ll_count, ll_most, nb_seq, LL_FSELOG, &PREDEFINED_LL_WEIGHTS, 6, true, strategy);
    let ll_coder = build_ctable(out, ll_type, &mut ll_count, &ll_codes, nb_seq, LL_FSELOG, &PREDEF_LL);

    let (mut of_count, of_most, of_max) = count_codes(&of_codes, MAX_OFF);
    let of_default_allowed = of_max <= DEFAULT_MAX_OFF;
    let of_type = select_encoding_type(&of_count, of_most, nb_seq, OFF_FSELOG, &PREDEFINED_OF_WEIGHTS, 5, of_default_allowed, strategy);
    let of_coder = build_ctable(out, of_type, &mut of_count, &of_codes, nb_seq, OFF_FSELOG, &PREDEF_OF);

    let (mut ml_count, ml_most, _) = count_codes(&ml_codes, MAX_ML);
    let ml_type = select_encoding_type(&ml_count, ml_most, nb_seq, ML_FSELOG, &PREDEFINED_ML_WEIGHTS, 6, true, strategy);
    let ml_coder = build_ctable(out, ml_type, &mut ml_count, &ml_codes, nb_seq, ML_FSELOG, &PREDEF_ML);

    out[seq_head] = ((ll_type as u8) << 6) | ((of_type as u8) << 4) | ((ml_type as u8) << 2);

    encode_sequences(out, seqs, &ll_codes, &ml_codes, &of_codes, ll_coder.table(), of_coder.table(), ml_coder.table());
    Some(())
}
