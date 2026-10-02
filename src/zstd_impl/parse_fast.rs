//! `ZSTD_compressBlock_fast_noDict_generic` and
//! `ZSTD_compressBlock_doubleFast_noDict_generic` (C zstd 1.5.7), ported
//! with the same control flow. Positions are `usize` indices into
//! `ctx.base` (the whole input so far); the hash tables persist across
//! blocks of one frame.
//!
//! All byte and table access goes through the unchecked helpers in
//! `seqstore`. Each parser states the invariants that make them sound once
//! in a `SAFETY` paragraph and tags every use site with the invariant it
//! relies on.

use super::seqstore::{byte, count, hash_at, offset_to_offbase, read32, read64, tbl_get, tbl_set, SeqStore, REPCODE1_TO_OFFBASE};

const K_SEARCH_STRENGTH: u32 = 8;
const HASH_READ_SIZE: usize = 8;

/// Match-finder state shared by the fast/dfast parsers.
pub struct FastState {
    /// `hashTable` (fast: the only table; dfast: the long 8-byte table).
    pub hash_table: Vec<u32>,
    /// dfast `chainTable` used as the short hash table.
    pub hash_small: Vec<u32>,
}

impl FastState {
    pub fn new() -> Self {
        FastState { hash_table: Vec::new(), hash_small: Vec::new() }
    }
    pub fn reset(&mut self, hash_log: u32, chain_log: u32, dfast: bool) {
        self.hash_table.clear();
        self.hash_table.resize(1 << hash_log, 0);
        self.hash_small.clear();
        if dfast {
            self.hash_small.resize(1 << chain_log, 0);
        }
    }
}

/// Block context handed to the parsers.
pub struct BlockCtx<'a> {
    /// The whole addressable input: history, the block, and whatever may be
    /// read past it. `base.len()` is the literal-copy / match-count limit.
    pub base: &'a [u8],
    /// First byte of the block (index into `base`).
    pub istart: u32,
    pub block_len: usize,
    pub window_log: u32,
    pub target_length: u32,
    pub rep: &'a mut [u32; 3],
}

impl BlockCtx<'_> {
    /// The geometry every parser relies on: the block lies inside `base`.
    #[inline(always)]
    pub(super) fn check(&self) {
        assert!(self.istart as usize + self.block_len <= self.base.len(), "zstd parser: block outside input");
    }
}

/// Returns the number of trailing literals (C returns `iend - anchor`).
///
/// # Safety
/// `st` was `reset` with this `hash_log` (so `hash_table.len() == 1 <<
/// hash_log`, which is asserted) and since then has only been fed prefixes
/// of `ctx.base` by this function, so every stored position is `<
/// ctx.base.len()`. `seq.reset(ctx.block_len)` was called for this block.
pub unsafe fn compress_block_fast(st: &mut FastState, seq: &mut SeqStore, ctx: &mut BlockCtx, hash_log: u32, mls: u32) -> usize {
    ctx.check();
    assert_eq!(st.hash_table.len(), 1usize << hash_log, "zstd fast: hash table not reset for hash_log");
    // SAFETY: the caller's contract, plus the two checks above.
    unsafe {
        match mls {
            4 => fast_generic::<4>(st, seq, ctx, hash_log),
            5 => fast_generic::<5>(st, seq, ctx, hash_log),
            6 => fast_generic::<6>(st, seq, ctx, hash_log),
            _ => fast_generic::<7>(st, seq, ctx, hash_log),
        }
    }
}

/// # Safety
/// `ip + 4 <= base.len()` and `match_idx + 4 <= base.len()`.
#[inline(always)]
unsafe fn match4_found(base: &[u8], ip: usize, match_idx: u32, idx_low: u32) -> bool {
    // SAFETY: per the contract above.
    unsafe { (read32(base, ip) == read32(base, match_idx as usize)) & (match_idx >= idx_low) }
}

/// # Safety
/// As [`compress_block_fast`].
#[inline(never)]
unsafe fn fast_generic<const MLS: u32>(st: &mut FastState, seq: &mut SeqStore, ctx: &mut BlockCtx, hlog: u32) -> usize {
    let hash_table = &mut st.hash_table[..];
    let base = ctx.base;
    let istart = ctx.istart as usize;
    let iend = istart + ctx.block_len;
    let end_index = iend as u32;
    let max_distance = 1u32 << ctx.window_log;
    let prefix_start_index = if end_index > max_distance { end_index - max_distance } else { 0 };
    let prefix_start = prefix_start_index as usize;
    // C computes `iend - HASH_READ_SIZE`, which may lie before `istart` for
    // a tiny block; saturating keeps every `>= ilimit` exit below equivalent
    // (the positions compared are all >= 1).
    let ilimit = iend.saturating_sub(HASH_READ_SIZE);
    let step_size = (ctx.target_length + (ctx.target_length == 0) as u32 + 1) as usize;
    let k_step_incr = 1usize << (K_SEARCH_STRENGTH - 1);

    let mut anchor = istart;
    let mut ip0 = istart;
    let mut rep_offset1 = ctx.rep[0];
    let mut rep_offset2 = ctx.rep[1];
    let mut offset_saved1 = 0u32;
    let mut offset_saved2 = 0u32;

    ip0 += (ip0 == prefix_start) as usize;
    {
        let curr = ip0 as u32;
        let window_low = if curr > max_distance { curr - max_distance } else { 0 };
        let max_rep = curr - window_low;
        if rep_offset2 > max_rep {
            offset_saved2 = rep_offset2;
            rep_offset2 = 0;
        }
        if rep_offset1 > max_rep {
            offset_saved1 = rep_offset1;
            rep_offset1 = 0;
        }
    }

    // SAFETY invariants for the loop below (`iend <= base.len()` by `check`):
    //  (a) ip0 < ip1 < ip2 < ip3 < ilimit = iend - 8, so 8 bytes are
    //      readable at any of them; after a match, `ip0 <= ilimit` is
    //      re-checked before `ip0 - 2`, `current0 + 2 <= ip0` are hashed.
    //  (b) rep_offset1/2 <= the current position: validated against
    //      `max_rep` above, afterwards only ever set to `ip0 - match0` with
    //      `match0 >= prefix_start`. So `ipN - rep` is in range (and the
    //      read is masked by `rep > 0` when rep is 0).
    //  (c) table entries are positions this parser stored for a prefix of
    //      `base` (caller contract), all `<= ilimit + 1`, so 4 bytes are
    //      readable there; table indices come from `hash_at` with `hlog`
    //      bits and the table has `1 << hlog` entries (asserted).
    //  (d) `count` is called with `match0 < ip0 <= iend`.
    macro_rules! hash {
        ($p:expr) => {
            // SAFETY: (a)
            unsafe { hash_at::<MLS>(base, $p, hlog) }
        };
    }

    'start: loop {
        let mut step = step_size;
        let mut next_step = ip0 + k_step_incr;
        let mut ip1 = ip0 + 1;
        let mut ip2 = ip0 + step;
        let mut ip3 = ip2 + 1;
        if ip3 >= ilimit {
            break 'start;
        }
        let mut hash0 = hash!(ip0);
        let mut hash1 = hash!(ip1);
        // SAFETY: (c)
        let mut match_idx = unsafe { tbl_get(hash_table, hash0) };
        let mut current0;

        // Outcome of the search loop.
        let mut m_length: usize;
        let off_base: u32;
        let match0: usize;
        loop {
            // SAFETY: (a), (b)
            let rval = unsafe { read32(base, ip2 - rep_offset1 as usize) };
            current0 = ip0 as u32;
            // SAFETY: (c)
            unsafe { tbl_set(hash_table, hash0, current0) };

            // SAFETY: (a)
            if (unsafe { read32(base, ip2) } == rval) & (rep_offset1 > 0) {
                ip0 = ip2;
                let mut m0 = ip0 - rep_offset1 as usize;
                // SAFETY: (a), (b): ip0 - 1 and m0 - 1 are both >= 0 since
                // ip0 >= 1 and rep_offset1 < ip0 (the match is 4 bytes long).
                let back = unsafe { (byte(base, ip0 - 1) == byte(base, m0 - 1)) as usize };
                ip0 -= back;
                m0 -= back;
                m_length = back + 4;
                // SAFETY: (c)
                unsafe { tbl_set(hash_table, hash1, ip1 as u32) };
                off_base = REPCODE1_TO_OFFBASE;
                match0 = m0;
                break;
            }
            // SAFETY: (a), (c)
            if unsafe { match4_found(base, ip0, match_idx, prefix_start_index) } {
                // SAFETY: (c)
                unsafe { tbl_set(hash_table, hash1, ip1 as u32) };
                // SAFETY: (a), (c)
                let (mo, ml, ob) = unsafe { fast_offset(base, ip0, match_idx, anchor, prefix_start, &mut rep_offset1, &mut rep_offset2) };
                ip0 = mo.0;
                match0 = mo.1;
                m_length = ml;
                off_base = ob;
                break;
            }
            // SAFETY: (c)
            match_idx = unsafe { tbl_get(hash_table, hash1) };
            hash0 = hash1;
            hash1 = hash!(ip2);
            ip0 = ip1;
            ip1 = ip2;
            ip2 = ip3;
            current0 = ip0 as u32;
            // SAFETY: (c)
            unsafe { tbl_set(hash_table, hash0, current0) };

            // SAFETY: (a), (c)
            if unsafe { match4_found(base, ip0, match_idx, prefix_start_index) } {
                if step <= 4 {
                    // SAFETY: (c)
                    unsafe { tbl_set(hash_table, hash1, ip1 as u32) };
                }
                // SAFETY: (a), (c)
                let (mo, ml, ob) = unsafe { fast_offset(base, ip0, match_idx, anchor, prefix_start, &mut rep_offset1, &mut rep_offset2) };
                ip0 = mo.0;
                match0 = mo.1;
                m_length = ml;
                off_base = ob;
                break;
            }
            // SAFETY: (c)
            match_idx = unsafe { tbl_get(hash_table, hash1) };
            hash0 = hash1;
            hash1 = hash!(ip2);
            ip0 = ip1;
            ip1 = ip2;
            ip2 = ip0 + step;
            ip3 = ip1 + step;
            if ip2 >= next_step {
                step += 1;
                next_step += k_step_incr;
            }
            if ip3 >= ilimit {
                break 'start;
            }
        }

        // _match: count the forward length and store.
        // SAFETY: (d)
        m_length += unsafe { count(base, ip0 + m_length, match0 + m_length, iend) };
        // SAFETY: the literals lie in `base` (anchor <= position <= iend <= base.len()) and `seq.reset(block_len)` covers this block.
        unsafe { seq.store_seq(anchor, ip0 - anchor, base, off_base, m_length) };
        ip0 += m_length;
        anchor = ip0;

        if ip0 <= ilimit {
            let p2 = current0 as usize + 2;
            // SAFETY: (a), (c)
            unsafe {
                tbl_set(hash_table, hash_at::<MLS>(base, p2, hlog), current0 + 2);
                tbl_set(hash_table, hash_at::<MLS>(base, ip0 - 2, hlog), ip0 as u32 - 2);
            }
            if rep_offset2 > 0 {
                // SAFETY: (a), (b)
                while ip0 <= ilimit && unsafe { read32(base, ip0) == read32(base, ip0 - rep_offset2 as usize) } {
                    // SAFETY: (d)
                    let r_length = unsafe { count(base, ip0 + 4, ip0 + 4 - rep_offset2 as usize, iend) } + 4;
                    core::mem::swap(&mut rep_offset1, &mut rep_offset2);
                    // SAFETY: (a), (c)
                    unsafe { tbl_set(hash_table, hash_at::<MLS>(base, ip0, hlog), ip0 as u32) };
                    ip0 += r_length;
                    // SAFETY: the literals lie in `base` (anchor <= position <= iend <= base.len()) and `seq.reset(block_len)` covers this block.
                    unsafe { seq.store_seq(anchor, 0, base, REPCODE1_TO_OFFBASE, r_length) };
                    anchor = ip0;
                }
            }
        }
    }

    offset_saved2 = if offset_saved1 != 0 && rep_offset1 != 0 { offset_saved1 } else { offset_saved2 };
    ctx.rep[0] = if rep_offset1 != 0 { rep_offset1 } else { offset_saved1 };
    ctx.rep[1] = if rep_offset2 != 0 { rep_offset2 } else { offset_saved2 };
    iend - anchor
}

/// `_offset:` label of the fast parser — compute offset, walk back.
///
/// # Safety
/// `match_idx < ip0 <= base.len()`.
#[inline(always)]
unsafe fn fast_offset(
    base: &[u8],
    mut ip0: usize,
    match_idx: u32,
    anchor: usize,
    prefix_start: usize,
    rep_offset1: &mut u32,
    rep_offset2: &mut u32,
) -> ((usize, usize), usize, u32) {
    let mut match0 = match_idx as usize;
    *rep_offset2 = *rep_offset1;
    *rep_offset1 = (ip0 - match0) as u32;
    let off_base = offset_to_offbase(*rep_offset1);
    let mut m_length = 4usize;
    // SAFETY: `ip0 > anchor >= 0` and `match0 > prefix_start >= 0` keep both
    // `- 1` positions in range.
    while ((ip0 > anchor) & (match0 > prefix_start)) && unsafe { byte(base, ip0 - 1) == byte(base, match0 - 1) } {
        ip0 -= 1;
        match0 -= 1;
        m_length += 1;
    }
    ((ip0, match0), m_length, off_base)
}

// ===========================================================================
// doubleFast
// ===========================================================================

/// # Safety
/// As [`compress_block_fast`], with `hash_small.len() == 1 << chain_log`
/// (asserted) as well.
pub unsafe fn compress_block_dfast(st: &mut FastState, seq: &mut SeqStore, ctx: &mut BlockCtx, hash_log: u32, chain_log: u32, mls: u32) -> usize {
    ctx.check();
    assert_eq!(st.hash_table.len(), 1usize << hash_log, "zstd dfast: long table not reset for hash_log");
    assert_eq!(st.hash_small.len(), 1usize << chain_log, "zstd dfast: short table not reset for chain_log");
    // SAFETY: the caller's contract, plus the checks above.
    unsafe {
        match mls {
            4 => dfast_generic::<4>(st, seq, ctx, hash_log, chain_log),
            5 => dfast_generic::<5>(st, seq, ctx, hash_log, chain_log),
            6 => dfast_generic::<6>(st, seq, ctx, hash_log, chain_log),
            _ => dfast_generic::<7>(st, seq, ctx, hash_log, chain_log),
        }
    }
}

/// Stand-in read target for a table entry below the window (C's `dummy[]`):
/// the compare happens unconditionally, the result is masked afterwards.
static DUMMY: [u8; 16] = [0x12, 0x34, 0x56, 0x78, 0x9a, 0xbc, 0xde, 0xf0, 0xe2, 0xb4, 0, 0, 0, 0, 0, 0];

/// # Safety
/// As [`compress_block_dfast`].
#[inline(never)]
unsafe fn dfast_generic<const MLS: u32>(st: &mut FastState, seq: &mut SeqStore, ctx: &mut BlockCtx, h_bits_l: u32, h_bits_s: u32) -> usize {
    let hash_long = &mut st.hash_table[..];
    let hash_small = &mut st.hash_small[..];
    let base = ctx.base;
    let istart = ctx.istart as usize;
    let iend = istart + ctx.block_len;
    let end_index = iend as u32;
    let max_distance = 1u32 << ctx.window_log;
    let prefix_lowest_index = if end_index > max_distance { end_index - max_distance } else { 0 };
    let prefix_lowest = prefix_lowest_index as usize;
    // See `fast_generic` for why this saturates.
    let ilimit = iend.saturating_sub(HASH_READ_SIZE);
    let k_step_incr = 1usize << K_SEARCH_STRENGTH;

    let mut anchor = istart;
    let mut ip = istart;
    let mut offset_1 = ctx.rep[0];
    let mut offset_2 = ctx.rep[1];
    let mut offset_saved1 = 0u32;
    let mut offset_saved2 = 0u32;

    ip += (ip == prefix_lowest) as usize;
    {
        let current = ip as u32;
        let window_low = if current > max_distance { current - max_distance } else { 0 };
        let max_rep = current - window_low;
        if offset_2 > max_rep {
            offset_saved2 = offset_2;
            offset_2 = 0;
        }
        if offset_1 > max_rep {
            offset_saved1 = offset_1;
            offset_1 = 0;
        }
    }

    // SAFETY invariants for the loop below (`iend <= base.len()` by `check`):
    //  (a) ip < ip1 <= ilimit = iend - 8, so 8 bytes are readable at both
    //      (and at `ip + 1`); after a match `ip <= ilimit` is re-checked
    //      before `ip - 1`, `ip - 2`, `curr + 2 <= ip` are hashed.
    //  (b) offset_1/2 <= the current position: validated against `max_rep`
    //      above, afterwards only set to `ip - match` for a verified match
    //      `>= prefix_lowest`. Reads at `ip - offset` are masked by
    //      `offset > 0`.
    //  (c) table entries are positions this parser stored for a prefix of
    //      `base` (caller contract), all `<= ilimit`, so 8 bytes are
    //      readable there; indices come from `hash_at` with `h_bits_l` /
    //      `h_bits_s` bits and the tables have that many entries (asserted).
    //  (d) `count` is called with `match < ip <= iend`.
    macro_rules! hl {
        ($p:expr) => {
            // SAFETY: (a)
            unsafe { hash_at::<8>(base, $p, h_bits_l) }
        };
    }
    macro_rules! hs {
        ($p:expr) => {
            // SAFETY: (a)
            unsafe { hash_at::<MLS>(base, $p, h_bits_s) }
        };
    }

    'outer: loop {
        let mut step = 1usize;
        let mut next_step = ip + k_step_incr;
        let mut ip1 = ip + step;
        if ip1 > ilimit {
            break 'outer;
        }
        let mut hl0 = hl!(ip);
        // SAFETY: (c)
        let mut idxl0 = unsafe { tbl_get(hash_long, hl0) };
        let mut matchl0 = idxl0 as usize;
        let mut curr: u32;

        let mut m_length: usize;
        let mut offset: u32;
        let hl1: usize;
        // Inner loop: one iteration per position searched.
        let stored_rep: bool;
        loop {
            let hs0 = hs!(ip);
            // SAFETY: (c)
            let idxs0 = unsafe { tbl_get(hash_small, hs0) };
            curr = ip as u32;
            let matchs0 = idxs0 as usize;
            // SAFETY: (c)
            unsafe {
                tbl_set(hash_long, hl0, curr);
                tbl_set(hash_small, hs0, curr);
            }

            // repcode at ip+1
            // SAFETY: (a), (b)
            if (offset_1 > 0) & unsafe { read32(base, ip + 1 - offset_1 as usize) == read32(base, ip + 1) } {
                // SAFETY: (d)
                m_length = unsafe { count(base, ip + 5, ip + 5 - offset_1 as usize, iend) } + 4;
                ip += 1;
                // SAFETY: the literals lie in `base` (anchor <= position <= iend <= base.len()) and `seq.reset(block_len)` covers this block.
                unsafe { seq.store_seq(anchor, ip - anchor, base, REPCODE1_TO_OFFBASE, m_length) };
                stored_rep = true;
                hl1 = 0;
                offset = 0;
                break;
            }

            let hl1_ = hl!(ip1);
            // An entry below the window is compared against DUMMY instead
            // (branch-free in C); the `&& valid` afterwards discards it.
            let l0_valid = idxl0 > prefix_lowest_index;
            let (l0_buf, l0_pos) = if l0_valid { (base, matchl0) } else { (&DUMMY[..], 0) };
            // SAFETY: (a), (c); DUMMY has 16 bytes.
            if unsafe { read64(l0_buf, l0_pos) == read64(base, ip) } && l0_valid {
                // SAFETY: (d)
                m_length = unsafe { count(base, ip + 8, matchl0 + 8, iend) } + 8;
                offset = (ip - matchl0) as u32;
                // SAFETY: `ip > anchor >= 0`, `matchl0 > prefix_lowest >= 0`.
                while ((ip > anchor) & (matchl0 > prefix_lowest)) && unsafe { byte(base, ip - 1) == byte(base, matchl0 - 1) } {
                    ip -= 1;
                    matchl0 -= 1;
                    m_length += 1;
                }
                hl1 = hl1_;
                stored_rep = false;
                break;
            }

            // SAFETY: (c)
            let idxl1 = unsafe { tbl_get(hash_long, hl1_) };
            let matchl1 = idxl1 as usize;
            let s0_valid = idxs0 > prefix_lowest_index;
            let (s0_buf, s0_pos) = if s0_valid { (base, matchs0) } else { (&DUMMY[..], 0) };
            // SAFETY: (a), (c); DUMMY has 16 bytes.
            if unsafe { read32(s0_buf, s0_pos) == read32(base, ip) } && s0_valid {
                // _search_next_long
                let mut ms0 = matchs0;
                // SAFETY: (d)
                m_length = unsafe { count(base, ip + 4, ms0 + 4, iend) } + 4;
                offset = (ip - ms0) as u32;
                // SAFETY: (a), (c)
                if idxl1 > prefix_lowest_index && unsafe { read64(base, matchl1) == read64(base, ip1) } {
                    // SAFETY: (d)
                    let l1len = unsafe { count(base, ip1 + 8, matchl1 + 8, iend) } + 8;
                    if l1len > m_length {
                        ip = ip1;
                        m_length = l1len;
                        offset = (ip - matchl1) as u32;
                        ms0 = matchl1;
                    }
                }
                // SAFETY: `ip > anchor >= 0`, `ms0 > prefix_lowest >= 0`.
                while ((ip > anchor) & (ms0 > prefix_lowest)) && unsafe { byte(base, ip - 1) == byte(base, ms0 - 1) } {
                    ip -= 1;
                    ms0 -= 1;
                    m_length += 1;
                }
                hl1 = hl1_;
                stored_rep = false;
                break;
            }

            if ip1 >= next_step {
                step += 1;
                next_step += k_step_incr;
            }
            ip = ip1;
            ip1 += step;
            hl0 = hl1_;
            idxl0 = idxl1;
            matchl0 = matchl1;
            if ip1 > ilimit {
                break 'outer;
            }
        }

        if !stored_rep {
            // _match_found
            offset_2 = offset_1;
            offset_1 = offset;
            if step < 4 {
                // SAFETY: (c)
                unsafe { tbl_set(hash_long, hl1, ip1 as u32) };
            }
            // SAFETY: the literals lie in `base` (anchor <= position <= iend <= base.len()) and `seq.reset(block_len)` covers this block.
            unsafe { seq.store_seq(anchor, ip - anchor, base, offset_to_offbase(offset), m_length) };
        }

        // _match_stored
        ip += m_length;
        anchor = ip;
        if ip <= ilimit {
            let index_to_insert = curr + 2;
            let p2 = index_to_insert as usize;
            // SAFETY: (a), (c)
            unsafe {
                tbl_set(hash_long, hash_at::<8>(base, p2, h_bits_l), index_to_insert);
                tbl_set(hash_long, hash_at::<8>(base, ip - 2, h_bits_l), ip as u32 - 2);
                tbl_set(hash_small, hash_at::<MLS>(base, p2, h_bits_s), index_to_insert);
                tbl_set(hash_small, hash_at::<MLS>(base, ip - 1, h_bits_s), ip as u32 - 1);
            }
            // SAFETY: (a), (b)
            while ip <= ilimit && (offset_2 > 0) & unsafe { read32(base, ip) == read32(base, ip - offset_2 as usize) } {
                // SAFETY: (d)
                let r_length = unsafe { count(base, ip + 4, ip + 4 - offset_2 as usize, iend) } + 4;
                core::mem::swap(&mut offset_1, &mut offset_2);
                // SAFETY: (a), (c)
                unsafe {
                    tbl_set(hash_small, hash_at::<MLS>(base, ip, h_bits_s), ip as u32);
                    tbl_set(hash_long, hash_at::<8>(base, ip, h_bits_l), ip as u32);
                }
                // SAFETY: the literals lie in `base` (anchor <= position <= iend <= base.len()) and `seq.reset(block_len)` covers this block.
                unsafe { seq.store_seq(anchor, 0, base, REPCODE1_TO_OFFBASE, r_length) };
                ip += r_length;
                anchor = ip;
            }
        }
    }

    offset_saved2 = if offset_saved1 != 0 && offset_1 != 0 { offset_saved1 } else { offset_saved2 };
    ctx.rep[0] = if offset_1 != 0 { offset_1 } else { offset_saved1 };
    ctx.rep[1] = if offset_2 != 0 { offset_2 } else { offset_saved2 };
    iend - anchor
}
