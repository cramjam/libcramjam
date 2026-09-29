//! Huffman tree construction and block emission — a port of zlib's
//! `trees.c` (`build_tree` / `gen_bitlen` / `gen_codes`, `scan_tree` /
//! `send_tree`, `_tr_flush_block`). Kept structurally identical to zlib so
//! the produced streams are the same size class as zlib/miniz at every
//! level, and every emitted code is complete (Kraft sum == 1) — strict
//! inflaters reject anything else.

use super::bitwriter::BitWriter;

pub const MAX_BITS: usize = 15;
pub const MAX_BL_BITS: usize = 7;
pub const LITERALS: usize = 256;
pub const END_BLOCK: usize = 256;
pub const LENGTH_CODES: usize = 29;
pub const L_CODES: usize = LITERALS + 1 + LENGTH_CODES; // 286
pub const D_CODES: usize = 30;
pub const BL_CODES: usize = 19;
pub const HEAP_SIZE: usize = 2 * L_CODES + 1;
const REP_3_6: usize = 16;
const REPZ_3_10: usize = 17;
const REPZ_11_138: usize = 18;

pub const EXTRA_LBITS: [u8; LENGTH_CODES] =
    [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4, 5, 5, 5, 5, 0];
pub const EXTRA_DBITS: [u8; D_CODES] =
    [0, 0, 0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 8, 8, 9, 9, 10, 10, 11, 11, 12, 12, 13, 13];
const EXTRA_BLBITS: [u8; BL_CODES] = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 3, 7];
pub const BL_ORDER: [usize; BL_CODES] = [16, 17, 18, 0, 8, 7, 9, 6, 10, 5, 11, 4, 12, 3, 13, 2, 14, 1, 15];

/// `base_length[code]`: first length of each length code (length - 3).
pub const BASE_LENGTH: [u16; LENGTH_CODES] = [
    0, 1, 2, 3, 4, 5, 6, 7, 8, 10, 12, 14, 16, 20, 24, 28, 32, 40, 48, 56, 64, 80, 96, 112, 128,
    160, 192, 224, 0,
];
/// `base_dist[code]`: first distance of each distance code (distance - 1).
pub const BASE_DIST: [u16; D_CODES] = [
    0, 1, 2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 64, 96, 128, 192, 256, 384, 512, 768, 1024, 1536,
    2048, 3072, 4096, 6144, 8192, 12288, 16384, 24576,
];

/// `_length_code[len - 3]`.
pub static LENGTH_CODE: [u8; 256] = {
    let mut t = [0u8; 256];
    let mut code = 0usize;
    let mut length = 0usize;
    while code < LENGTH_CODES - 1 {
        let n = 1usize << EXTRA_LBITS[code];
        let mut i = 0;
        while i < n {
            t[length] = code as u8;
            length += 1;
            i += 1;
        }
        code += 1;
    }
    // Length 258 (index 255) uses code 28.
    t[255] = 28;
    t
};

/// `_dist_code[]`: 0..256 direct for dist-1 < 256, 256.. indexed by
/// `256 + ((dist-1) >> 7)`.
pub static DIST_CODE: [u8; 512] = {
    let mut t = [0u8; 512];
    let mut code = 0usize;
    let mut dist = 0usize;
    while code < 16 {
        let n = 1usize << EXTRA_DBITS[code];
        let mut i = 0;
        while i < n {
            t[dist] = code as u8;
            dist += 1;
            i += 1;
        }
        code += 1;
    }
    dist >>= 7;
    while code < D_CODES {
        let n = 1usize << (EXTRA_DBITS[code] - 7);
        let mut i = 0;
        while i < n {
            t[256 + dist] = code as u8;
            dist += 1;
            i += 1;
        }
        code += 1;
    }
    t
};

#[inline(always)]
pub fn d_code(dist_minus_1: usize) -> usize {
    if dist_minus_1 < 256 {
        DIST_CODE[dist_minus_1] as usize
    } else {
        DIST_CODE[256 + (dist_minus_1 >> 7)] as usize
    }
}

/// One tree node: frequency (or code after `gen_codes`) and code length
/// (or parent index during construction) — zlib's `ct_data` union.
#[derive(Clone, Copy, Default)]
pub struct Node {
    pub fc: u16, // freq, then code
    pub dl: u16, // dad, then len
}

pub struct StaticTrees {
    pub ltree: [Node; L_CODES + 2],
    pub dtree: [Node; D_CODES],
}

pub fn static_trees() -> &'static StaticTrees {
    static T: std::sync::OnceLock<StaticTrees> = std::sync::OnceLock::new();
    T.get_or_init(|| {
        let mut ltree = [Node::default(); L_CODES + 2];
        let mut bl_count = [0u16; MAX_BITS + 1];
        for (n, node) in ltree.iter_mut().enumerate() {
            node.dl = match n {
                0..=143 => 8,
                144..=255 => 9,
                256..=279 => 7,
                _ => 8,
            };
            bl_count[node.dl as usize] += 1;
        }
        gen_codes(&mut ltree, L_CODES + 1, &bl_count);
        let mut dtree = [Node::default(); D_CODES];
        for (n, node) in dtree.iter_mut().enumerate() {
            node.dl = 5;
            node.fc = bit_reverse(n as u32, 5) as u16;
        }
        StaticTrees { ltree, dtree }
    })
}

#[inline(always)]
fn bit_reverse(code: u32, len: u32) -> u32 {
    code.reverse_bits() >> (32 - len)
}

/// Generate canonical codes from the lengths in `tree[..max_code+1]`.
fn gen_codes(tree: &mut [Node], max_code: usize, bl_count: &[u16; MAX_BITS + 1]) {
    let mut next_code = [0u16; MAX_BITS + 1];
    let mut code = 0u16;
    for bits in 1..=MAX_BITS {
        code = (code + bl_count[bits - 1]) << 1;
        next_code[bits] = code;
    }
    for node in tree.iter_mut().take(max_code + 1) {
        let len = node.dl as u32;
        if len == 0 {
            continue;
        }
        node.fc = bit_reverse(next_code[len as usize] as u32, len) as u16;
        next_code[len as usize] += 1;
    }
}

/// Descriptor for one dynamic tree (zlib's `tree_desc` + `static_tree_desc`).
struct TreeDesc<'a> {
    stree: Option<&'a [Node]>,
    extra_bits: &'a [u8],
    extra_base: usize,
    elems: usize,
    max_length: usize,
}

/// Per-block Huffman state (zlib's tree-related `deflate_state` fields).
pub struct TreeState {
    pub dyn_ltree: [Node; HEAP_SIZE],
    pub dyn_dtree: [Node; 2 * D_CODES + 1],
    pub bl_tree: [Node; 2 * BL_CODES + 1],
    heap: [i32; 2 * L_CODES + 1],
    heap_len: usize,
    heap_max: usize,
    depth: [u8; 2 * L_CODES + 1],
    bl_count: [u16; MAX_BITS + 1],
    opt_len: u64,
    static_len: u64,
    l_max_code: usize,
    d_max_code: usize,
}

impl TreeState {
    pub fn new() -> Self {
        let mut s = Self {
            dyn_ltree: [Node::default(); HEAP_SIZE],
            dyn_dtree: [Node::default(); 2 * D_CODES + 1],
            bl_tree: [Node::default(); 2 * BL_CODES + 1],
            heap: [0; 2 * L_CODES + 1],
            heap_len: 0,
            heap_max: 0,
            depth: [0; 2 * L_CODES + 1],
            bl_count: [0; MAX_BITS + 1],
            opt_len: 0,
            static_len: 0,
            l_max_code: 0,
            d_max_code: 0,
        };
        s.init_block();
        s
    }

    pub fn init_block(&mut self) {
        for n in self.dyn_ltree.iter_mut().take(L_CODES) {
            n.fc = 0;
        }
        for n in self.dyn_dtree.iter_mut().take(D_CODES) {
            n.fc = 0;
        }
        for n in self.bl_tree.iter_mut().take(BL_CODES) {
            n.fc = 0;
        }
        self.dyn_ltree[END_BLOCK].fc = 1;
        self.opt_len = 0;
        self.static_len = 0;
    }

    #[inline(always)]
    fn smaller(tree: &[Node], n: usize, m: usize, depth: &[u8]) -> bool {
        tree[n].fc < tree[m].fc || (tree[n].fc == tree[m].fc && depth[n] <= depth[m])
    }

    fn pqdownheap(&mut self, tree: &[Node], mut k: usize) {
        let v = self.heap[k];
        let mut j = k << 1;
        while j <= self.heap_len {
            if j < self.heap_len
                && Self::smaller(tree, self.heap[j + 1] as usize, self.heap[j] as usize, &self.depth)
            {
                j += 1;
            }
            if Self::smaller(tree, v as usize, self.heap[j] as usize, &self.depth) {
                break;
            }
            self.heap[k] = self.heap[j];
            k = j;
            j <<= 1;
        }
        self.heap[k] = v;
    }

    /// zlib `gen_bitlen`: compute lengths from the tree shape, then repair
    /// overflow above `max_length` (this is what keeps codes complete).
    fn gen_bitlen(&mut self, tree: &mut [Node], desc: &TreeDesc, max_code: usize) {
        let max_length = desc.max_length;
        self.bl_count = [0; MAX_BITS + 1];
        let root = self.heap[self.heap_max] as usize;
        tree[root].dl = 0;
        let mut overflow: i32 = 0;
        let mut h = self.heap_max + 1;
        while h < HEAP_SIZE {
            let n = self.heap[h] as usize;
            let mut bits = tree[tree[n].dl as usize].dl as usize + 1;
            if bits > max_length {
                bits = max_length;
                overflow += 1;
            }
            tree[n].dl = bits as u16;
            h += 1;
            if n > max_code {
                continue; // not a leaf
            }
            self.bl_count[bits] += 1;
            let xbits = if n >= desc.extra_base { desc.extra_bits[n - desc.extra_base] as u64 } else { 0 };
            let f = tree[n].fc as u64;
            // zlib accumulates `opt_len`/`static_len` in a `ulg` and the
            // overflow-repair below can transiently drive it "negative"
            // (a shortened code subtracts more than was added) before it
            // nets back positive — modular arithmetic by design. Use
            // wrapping ops so a debug-assertions build doesn't panic on a
            // degenerate/adversarial tree (release wraps and still emits a
            // valid — if not size-optimal — block; the corpus tests pin
            // byte-identical output for real inputs).
            self.opt_len = self.opt_len.wrapping_add(f.wrapping_mul(bits as u64 + xbits));
            if let Some(stree) = desc.stree {
                self.static_len = self.static_len.wrapping_add(f.wrapping_mul(stree[n].dl as u64 + xbits));
            }
        }
        if overflow == 0 {
            return;
        }
        loop {
            let mut bits = max_length - 1;
            while bits > 0 && self.bl_count[bits] == 0 {
                bits -= 1;
            }
            if bits == 0 {
                // No shorter code to lengthen — degenerate tree; stop
                // repairing (matches zlib terminating; avoids usize underflow).
                break;
            }
            self.bl_count[bits] -= 1;
            self.bl_count[bits + 1] += 2;
            self.bl_count[max_length] -= 1;
            overflow -= 2;
            if overflow <= 0 {
                break;
            }
        }
        let mut h = HEAP_SIZE;
        let mut bits = max_length;
        while bits != 0 {
            let mut n = self.bl_count[bits];
            while n != 0 {
                h -= 1;
                let m = self.heap[h] as usize;
                if m > max_code {
                    continue;
                }
                if tree[m].dl as usize != bits {
                    self.opt_len = self
                        .opt_len
                        .wrapping_add((bits as i64 - tree[m].dl as i64).wrapping_mul(tree[m].fc as i64) as u64);
                    tree[m].dl = bits as u16;
                }
                n -= 1;
            }
            bits -= 1;
        }
    }

    /// zlib `build_tree`. Returns `max_code`.
    fn build_tree(&mut self, tree: &mut [Node], desc: &TreeDesc) -> usize {
        let elems = desc.elems;
        let mut max_code: i32 = -1;
        self.heap_len = 0;
        self.heap_max = HEAP_SIZE;
        for n in 0..elems {
            if tree[n].fc != 0 {
                self.heap_len += 1;
                self.heap[self.heap_len] = n as i32;
                max_code = n as i32;
                self.depth[n] = 0;
            } else {
                tree[n].dl = 0;
            }
        }
        while self.heap_len < 2 {
            let node = if max_code < 2 {
                max_code += 1;
                max_code as usize
            } else {
                0
            };
            self.heap_len += 1;
            self.heap[self.heap_len] = node as i32;
            tree[node].fc = 1;
            self.depth[node] = 0;
            self.opt_len = self.opt_len.wrapping_sub(1);
            if let Some(stree) = desc.stree {
                self.static_len = self.static_len.wrapping_sub(stree[node].dl as u64);
            }
        }
        let max_code = max_code as usize;
        let mut n = self.heap_len / 2;
        while n >= 1 {
            self.pqdownheap(tree, n);
            n -= 1;
        }
        let mut node = elems;
        loop {
            // pqremove
            let n = self.heap[1] as usize;
            self.heap[1] = self.heap[self.heap_len];
            self.heap_len -= 1;
            self.pqdownheap(tree, 1);
            let m = self.heap[1] as usize;
            self.heap_max -= 1;
            self.heap[self.heap_max] = n as i32;
            self.heap_max -= 1;
            self.heap[self.heap_max] = m as i32;
            // zlib sums `ush` (u16) frequencies here and lets them wrap
            // silently. Real blocks bound every frequency by the block size
            // (<= 16383 symbols), so the sum never overflows in practice and
            // this equals a plain add; `wrapping_add` matches zlib exactly and
            // avoids a debug-assertions overflow panic on synthetic/degenerate
            // frequency distributions (the tree stays a valid complete code —
            // fc only orders the priority queue).
            tree[node].fc = tree[n].fc.wrapping_add(tree[m].fc);
            self.depth[node] = self.depth[n].max(self.depth[m]) + 1;
            tree[n].dl = node as u16;
            tree[m].dl = node as u16;
            self.heap[1] = node as i32;
            node += 1;
            self.pqdownheap(tree, 1);
            if self.heap_len < 2 {
                break;
            }
        }
        self.heap_max -= 1;
        self.heap[self.heap_max] = self.heap[1];
        self.gen_bitlen(tree, desc, max_code);
        gen_codes(tree, max_code, &self.bl_count);
        max_code
    }

    /// zlib `scan_tree`: accumulate code-length-alphabet frequencies.
    fn scan_tree(&mut self, which: u8, max_code: usize) {
        let len_at = |s: &Self, i: usize| -> usize {
            let t: &[Node] = if which == 0 { &s.dyn_ltree } else { &s.dyn_dtree };
            if i > max_code { 0xffff } else { t[i].dl as usize }
        };
        let mut prevlen: usize = usize::MAX;
        let mut nextlen = len_at(self, 0);
        let mut count = 0usize;
        let mut max_count = 7usize;
        let mut min_count = 4usize;
        if nextlen == 0 {
            max_count = 138;
            min_count = 3;
        }
        for n in 0..=max_code {
            let curlen = nextlen;
            nextlen = len_at(self, n + 1);
            count += 1;
            if count < max_count && curlen == nextlen {
                continue;
            } else if count < min_count {
                self.bl_tree[curlen].fc += count as u16;
            } else if curlen != 0 {
                if curlen != prevlen {
                    self.bl_tree[curlen].fc += 1;
                }
                self.bl_tree[REP_3_6].fc += 1;
            } else if count <= 10 {
                self.bl_tree[REPZ_3_10].fc += 1;
            } else {
                self.bl_tree[REPZ_11_138].fc += 1;
            }
            count = 0;
            prevlen = curlen;
            if nextlen == 0 {
                max_count = 138;
                min_count = 3;
            } else if curlen == nextlen {
                max_count = 6;
                min_count = 3;
            } else {
                max_count = 7;
                min_count = 4;
            }
        }
    }

    /// zlib `send_tree`.
    fn send_tree(&self, w: &mut BitWriter, which: u8, max_code: usize) {
        let t: &[Node] = if which == 0 { &self.dyn_ltree } else { &self.dyn_dtree };
        let len_at = |i: usize| -> usize { if i > max_code { 0xffff } else { t[i].dl as usize } };
        let bl = &self.bl_tree;
        let send = |w: &mut BitWriter, c: usize| w.write_bits(bl[c].fc as u32, bl[c].dl as u32);
        let mut prevlen: usize = usize::MAX;
        let mut nextlen = len_at(0);
        let mut count = 0usize;
        let mut max_count = 7usize;
        let mut min_count = 4usize;
        if nextlen == 0 {
            max_count = 138;
            min_count = 3;
        }
        for n in 0..=max_code {
            let curlen = nextlen;
            nextlen = len_at(n + 1);
            count += 1;
            if count < max_count && curlen == nextlen {
                continue;
            } else if count < min_count {
                for _ in 0..count {
                    send(w, curlen);
                }
            } else if curlen != 0 {
                if curlen != prevlen {
                    send(w, curlen);
                    count -= 1;
                }
                send(w, REP_3_6);
                w.write_bits((count - 3) as u32, 2);
            } else if count <= 10 {
                send(w, REPZ_3_10);
                w.write_bits((count - 3) as u32, 3);
            } else {
                send(w, REPZ_11_138);
                w.write_bits((count - 11) as u32, 7);
            }
            count = 0;
            prevlen = curlen;
            if nextlen == 0 {
                max_count = 138;
                min_count = 3;
            } else if curlen == nextlen {
                max_count = 6;
                min_count = 3;
            } else {
                max_count = 7;
                min_count = 4;
            }
        }
    }

    /// zlib `build_bl_tree`: returns `max_blindex`.
    fn build_bl_tree(&mut self) -> usize {
        self.scan_tree(0, self.l_max_code);
        self.scan_tree(1, self.d_max_code);
        let mut bl = self.bl_tree;
        let desc = TreeDesc {
            stree: None,
            extra_bits: &EXTRA_BLBITS,
            extra_base: 0,
            elems: BL_CODES,
            max_length: MAX_BL_BITS,
        };
        self.build_tree(&mut bl, &desc);
        self.bl_tree = bl;
        let mut max_blindex = BL_CODES - 1;
        while max_blindex >= 3 {
            if self.bl_tree[BL_ORDER[max_blindex]].dl != 0 {
                break;
            }
            max_blindex -= 1;
        }
        self.opt_len += 3 * (max_blindex as u64 + 1) + 5 + 5 + 4;
        max_blindex
    }

    fn send_all_trees(&self, w: &mut BitWriter, lcodes: usize, dcodes: usize, blcodes: usize) {
        w.write_bits((lcodes - 257) as u32, 5);
        w.write_bits((dcodes - 1) as u32, 5);
        w.write_bits((blcodes - 4) as u32, 4);
        for &o in BL_ORDER.iter().take(blcodes) {
            w.write_bits(self.bl_tree[o].dl as u32, 3);
        }
        self.send_tree(w, 0, lcodes - 1);
        self.send_tree(w, 1, dcodes - 1);
    }

    /// zlib `_tr_flush_block`: choose stored / static / dynamic and emit.
    /// `syms` is the block's symbol buffer (`dist == 0` → literal `lc`),
    /// `stored` the raw bytes it covers.
    pub fn flush_block(&mut self, w: &mut BitWriter, syms: &[Sym], stored: &[u8], last: bool) {
        let st = static_trees();
        let mut ltree = self.dyn_ltree;
        let mut dtree = self.dyn_dtree;
        let ldesc = TreeDesc {
            stree: Some(&st.ltree),
            extra_bits: &EXTRA_LBITS,
            extra_base: LITERALS + 1,
            elems: L_CODES,
            max_length: MAX_BITS,
        };
        self.l_max_code = self.build_tree(&mut ltree, &ldesc);
        let ddesc = TreeDesc {
            stree: Some(&st.dtree),
            extra_bits: &EXTRA_DBITS,
            extra_base: 0,
            elems: D_CODES,
            max_length: MAX_BITS,
        };
        self.d_max_code = self.build_tree(&mut dtree, &ddesc);
        self.dyn_ltree = ltree;
        self.dyn_dtree = dtree;
        let max_blindex = self.build_bl_tree();
        let mut opt_lenb = (self.opt_len + 3 + 7) >> 3;
        let static_lenb = (self.static_len + 3 + 7) >> 3;
        if static_lenb <= opt_lenb {
            opt_lenb = static_lenb;
        }
        let stored_len = stored.len() as u64;
        if stored_len + 4 <= opt_lenb && stored_len <= 0xFFFF {
            w.write_bits(last as u32, 3); // BTYPE 00
            w.align_to_byte();
            w.write_u16_le(stored_len as u16);
            w.write_u16_le(!(stored_len as u16));
            w.write_bytes(stored);
        } else if static_lenb == opt_lenb {
            w.write_bits((1 << 1) | last as u32, 3);
            compress_block(w, syms, &st.ltree, &st.dtree);
        } else {
            w.write_bits((2 << 1) | last as u32, 3);
            self.send_all_trees(w, self.l_max_code + 1, self.d_max_code + 1, max_blindex + 1);
            compress_block(w, syms, &self.dyn_ltree, &self.dyn_dtree);
        }
        self.init_block();
    }
}

/// One LZ77 symbol as stored by the parser: `dist == 0` means literal `lc`,
/// otherwise a match of length `lc + 3` at distance `dist`.
#[derive(Clone, Copy)]
pub struct Sym {
    pub dist: u16,
    pub lc: u8,
}

/// zlib `compress_block`.
#[inline(never)]
fn compress_block(w: &mut BitWriter, syms: &[Sym], ltree: &[Node], dtree: &[Node]) {
    for s in syms {
        let dist = s.dist as usize;
        let lc = s.lc as usize;
        if dist == 0 {
            w.write_bits(ltree[lc].fc as u32, ltree[lc].dl as u32);
        } else {
            let code = LENGTH_CODE[lc] as usize;
            let n = ltree[code + LITERALS + 1];
            w.write_bits(n.fc as u32, n.dl as u32);
            let extra = EXTRA_LBITS[code] as u32;
            if extra != 0 {
                w.write_bits((lc - BASE_LENGTH[code] as usize) as u32, extra);
            }
            let d = dist - 1;
            let dc = d_code(d);
            let n = dtree[dc];
            w.write_bits(n.fc as u32, n.dl as u32);
            let extra = EXTRA_DBITS[dc] as u32;
            if extra != 0 {
                w.write_bits((d - BASE_DIST[dc] as usize) as u32, extra);
            }
        }
    }
    let n = ltree[END_BLOCK];
    w.write_bits(n.fc as u32, n.dl as u32);
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Every code `build_tree` produces must be complete (Kraft sum == 1)
    /// even when the depth limit rewrites lengths — strict inflaters
    /// (miniz, zlib) reject over-subscribed or incomplete codes.
    #[test]
    fn build_tree_is_complete_under_depth_limit() {
        let mut x = 0x2545_F491_4F6C_DD1Du64;
        let mut ts = TreeState::new();
        let st = static_trees();
        for round in 0..2000 {
            let elems = if round % 2 == 0 { L_CODES } else { BL_CODES };
            let max_length = if elems == L_CODES { MAX_BITS } else { MAX_BL_BITS };
            let mut tree = [Node::default(); HEAP_SIZE];
            let used = 2 + (round % (elems - 2));
            for n in 0..used {
                x ^= x << 13;
                x ^= x >> 7;
                x ^= x << 17;
                // Skewed frequencies so the limit kicks in often.
                tree[n].fc = ((1u64 << (x % 15)) as u16).max(1);
            }
            let desc = TreeDesc {
                stree: if elems == L_CODES { Some(&st.ltree) } else { None },
                extra_bits: if elems == L_CODES { &EXTRA_LBITS } else { &EXTRA_BLBITS },
                extra_base: if elems == L_CODES { LITERALS + 1 } else { 0 },
                elems,
                max_length,
            };
            ts.init_block();
            let max_code = ts.build_tree(&mut tree, &desc);
            let mut kraft = 0u64;
            for n in 0..=max_code {
                let l = tree[n].dl as u32;
                assert!(l as usize <= max_length, "length {l} > {max_length}");
                if l > 0 {
                    kraft += 1u64 << (max_length as u32 - l);
                }
            }
            assert_eq!(kraft, 1u64 << max_length, "Kraft sum != 1 (round {round})");
        }
    }
}
