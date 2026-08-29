//! `ZSTD_compressBlock_fast_noDict_generic` and
//! `ZSTD_compressBlock_doubleFast_noDict_generic` (C zstd 1.5.7), ported
//! with the same control flow. Positions are `u32` indices into the whole
//! input (`base`); the hash tables persist across blocks of one frame.
#![allow(unused_unsafe)]

use super::seqstore::{count, hash_ptr, offset_to_offbase, read32, read64, SeqStore, REPCODE1_TO_OFFBASE};

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
    pub base: *const u8,
    /// First byte of the block (index into `base`).
    pub istart: u32,
    pub block_len: usize,
    /// End of the whole input (literal-copy limit / count limit).
    pub input_end: *const u8,
    pub window_log: u32,
    pub target_length: u32,
    pub rep: &'a mut [u32; 3],
}

/// Returns the number of trailing literals (C returns `iend - anchor`).
pub unsafe fn compress_block_fast(st: &mut FastState, seq: &mut SeqStore, ctx: &mut BlockCtx, hash_log: u32, mls: u32) -> usize {
    match mls {
        4 => unsafe { fast_generic::<4>(st, seq, ctx, hash_log) },
        5 => unsafe { fast_generic::<5>(st, seq, ctx, hash_log) },
        6 => unsafe { fast_generic::<6>(st, seq, ctx, hash_log) },
        _ => unsafe { fast_generic::<7>(st, seq, ctx, hash_log) },
    }
}

#[inline(always)]
unsafe fn match4_found(ip: *const u8, m: *const u8, match_idx: u32, idx_low: u32) -> bool {
    unsafe { (read32(ip) == read32(m)) & (match_idx >= idx_low) }
}

#[inline(never)]
unsafe fn fast_generic<const MLS: u32>(st: &mut FastState, seq: &mut SeqStore, ctx: &mut BlockCtx, hlog: u32) -> usize {
    let hash_table = st.hash_table.as_mut_ptr();
    let base = ctx.base;
    let istart = unsafe { base.add(ctx.istart as usize) };
    let iend = unsafe { istart.add(ctx.block_len) };
    let end_index = ctx.istart + ctx.block_len as u32;
    let max_distance = 1u32 << ctx.window_log;
    let prefix_start_index = if end_index > max_distance { end_index - max_distance } else { 0 };
    let prefix_start = unsafe { base.add(prefix_start_index as usize) };
    let ilimit = unsafe { iend.sub(HASH_READ_SIZE) };
    let step_size = (ctx.target_length + (ctx.target_length == 0) as u32 + 1) as usize;
    let k_step_incr = 1usize << (K_SEARCH_STRENGTH - 1);
    let lit_limit = ctx.input_end;

    let mut anchor = istart;
    let mut ip0 = istart;
    let mut rep_offset1 = ctx.rep[0];
    let mut rep_offset2 = ctx.rep[1];
    let mut offset_saved1 = 0u32;
    let mut offset_saved2 = 0u32;

    ip0 = unsafe { ip0.add((ip0 == prefix_start) as usize) };
    {
        let curr = unsafe { ip0.offset_from(base) } as u32;
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

    macro_rules! hash {
        ($p:expr) => {
            unsafe { hash_ptr::<MLS>($p, hlog) }
        };
    }

    'start: loop {
        let mut step = step_size;
        let mut next_step = unsafe { ip0.add(k_step_incr) };
        let mut ip1 = unsafe { ip0.add(1) };
        let mut ip2 = unsafe { ip0.add(step) };
        let mut ip3 = unsafe { ip2.add(1) };
        if ip3 >= ilimit {
            break 'start;
        }
        let mut hash0 = hash!(ip0);
        let mut hash1 = hash!(ip1);
        let mut match_idx = unsafe { *hash_table.add(hash0) };
        let mut current0;

        // Outcome of the search loop.
        let mut m_length: usize;
        let off_base: u32;
        let match0: *const u8;
        loop {
            let rval = unsafe { read32(ip2.sub(rep_offset1 as usize)) };
            current0 = unsafe { ip0.offset_from(base) } as u32;
            unsafe { *hash_table.add(hash0) = current0 };

            if (unsafe { read32(ip2) } == rval) & (rep_offset1 > 0) {
                ip0 = ip2;
                let mut m0 = unsafe { ip0.sub(rep_offset1 as usize) };
                let back = unsafe { (*ip0.sub(1) == *m0.sub(1)) as usize };
                ip0 = unsafe { ip0.sub(back) };
                m0 = unsafe { m0.sub(back) };
                m_length = back + 4;
                unsafe { *hash_table.add(hash1) = ip1.offset_from(base) as u32 };
                off_base = REPCODE1_TO_OFFBASE;
                match0 = m0;
                break;
            }
            if unsafe { match4_found(ip0, base.add(match_idx as usize), match_idx, prefix_start_index) } {
                unsafe { *hash_table.add(hash1) = ip1.offset_from(base) as u32 };
                let (mo, ml, ob) = unsafe { fast_offset(ip0, base, match_idx, anchor, prefix_start, &mut rep_offset1, &mut rep_offset2) };
                ip0 = mo.0;
                match0 = mo.1;
                m_length = ml;
                off_base = ob;
                break;
            }
            match_idx = unsafe { *hash_table.add(hash1) };
            hash0 = hash1;
            hash1 = hash!(ip2);
            ip0 = ip1;
            ip1 = ip2;
            ip2 = ip3;
            current0 = unsafe { ip0.offset_from(base) } as u32;
            unsafe { *hash_table.add(hash0) = current0 };

            if unsafe { match4_found(ip0, base.add(match_idx as usize), match_idx, prefix_start_index) } {
                if step <= 4 {
                    unsafe { *hash_table.add(hash1) = ip1.offset_from(base) as u32 };
                }
                let (mo, ml, ob) = unsafe { fast_offset(ip0, base, match_idx, anchor, prefix_start, &mut rep_offset1, &mut rep_offset2) };
                ip0 = mo.0;
                match0 = mo.1;
                m_length = ml;
                off_base = ob;
                break;
            }
            match_idx = unsafe { *hash_table.add(hash1) };
            hash0 = hash1;
            hash1 = hash!(ip2);
            ip0 = ip1;
            ip1 = ip2;
            ip2 = unsafe { ip0.add(step) };
            ip3 = unsafe { ip1.add(step) };
            if ip2 >= next_step {
                step += 1;
                next_step = unsafe { next_step.add(k_step_incr) };
            }
            if ip3 >= ilimit {
                break 'start;
            }
        }

        // _match: count the forward length and store.
        m_length += unsafe { count(ip0.add(m_length), match0.add(m_length), iend) };
        unsafe { seq.store_seq(ip0.offset_from(anchor) as usize, anchor, lit_limit, off_base, m_length) };
        ip0 = unsafe { ip0.add(m_length) };
        anchor = ip0;

        if ip0 <= ilimit {
            unsafe {
                let p2 = base.add(current0 as usize + 2);
                *hash_table.add(hash!(p2)) = current0 + 2;
                *hash_table.add(hash!(ip0.sub(2))) = ip0.offset_from(base) as u32 - 2;
            }
            if rep_offset2 > 0 {
                while ip0 <= ilimit && unsafe { read32(ip0) == read32(ip0.sub(rep_offset2 as usize)) } {
                    let r_length = unsafe { count(ip0.add(4), ip0.add(4).sub(rep_offset2 as usize), iend) } + 4;
                    core::mem::swap(&mut rep_offset1, &mut rep_offset2);
                    unsafe { *hash_table.add(hash!(ip0)) = ip0.offset_from(base) as u32 };
                    ip0 = unsafe { ip0.add(r_length) };
                    unsafe { seq.store_seq(0, anchor, lit_limit, REPCODE1_TO_OFFBASE, r_length) };
                    anchor = ip0;
                }
            }
        }
    }

    offset_saved2 = if offset_saved1 != 0 && rep_offset1 != 0 { offset_saved1 } else { offset_saved2 };
    ctx.rep[0] = if rep_offset1 != 0 { rep_offset1 } else { offset_saved1 };
    ctx.rep[1] = if rep_offset2 != 0 { rep_offset2 } else { offset_saved2 };
    unsafe { iend.offset_from(anchor) as usize }
}

/// `_offset:` label of the fast parser — compute offset, walk back.
#[inline(always)]
unsafe fn fast_offset(
    mut ip0: *const u8,
    base: *const u8,
    match_idx: u32,
    anchor: *const u8,
    prefix_start: *const u8,
    rep_offset1: &mut u32,
    rep_offset2: &mut u32,
) -> ((*const u8, *const u8), usize, u32) {
    let mut match0 = unsafe { base.add(match_idx as usize) };
    *rep_offset2 = *rep_offset1;
    *rep_offset1 = unsafe { ip0.offset_from(match0) } as u32;
    let off_base = offset_to_offbase(*rep_offset1);
    let mut m_length = 4usize;
    while ((ip0 > anchor) & (match0 > prefix_start)) && unsafe { *ip0.sub(1) == *match0.sub(1) } {
        ip0 = unsafe { ip0.sub(1) };
        match0 = unsafe { match0.sub(1) };
        m_length += 1;
    }
    ((ip0, match0), m_length, off_base)
}

// ===========================================================================
// doubleFast
// ===========================================================================

pub unsafe fn compress_block_dfast(st: &mut FastState, seq: &mut SeqStore, ctx: &mut BlockCtx, hash_log: u32, chain_log: u32, mls: u32) -> usize {
    match mls {
        4 => unsafe { dfast_generic::<4>(st, seq, ctx, hash_log, chain_log) },
        5 => unsafe { dfast_generic::<5>(st, seq, ctx, hash_log, chain_log) },
        6 => unsafe { dfast_generic::<6>(st, seq, ctx, hash_log, chain_log) },
        _ => unsafe { dfast_generic::<7>(st, seq, ctx, hash_log, chain_log) },
    }
}

static DUMMY: [u8; 16] = [0x12, 0x34, 0x56, 0x78, 0x9a, 0xbc, 0xde, 0xf0, 0xe2, 0xb4, 0, 0, 0, 0, 0, 0];

#[inline(never)]
unsafe fn dfast_generic<const MLS: u32>(st: &mut FastState, seq: &mut SeqStore, ctx: &mut BlockCtx, h_bits_l: u32, h_bits_s: u32) -> usize {
    let hash_long = st.hash_table.as_mut_ptr();
    let hash_small = st.hash_small.as_mut_ptr();
    let base = ctx.base;
    let istart = unsafe { base.add(ctx.istart as usize) };
    let iend = unsafe { istart.add(ctx.block_len) };
    let end_index = ctx.istart + ctx.block_len as u32;
    let max_distance = 1u32 << ctx.window_log;
    let prefix_lowest_index = if end_index > max_distance { end_index - max_distance } else { 0 };
    let prefix_lowest = unsafe { base.add(prefix_lowest_index as usize) };
    let ilimit = unsafe { iend.sub(HASH_READ_SIZE) };
    let k_step_incr = 1usize << K_SEARCH_STRENGTH;
    let lit_limit = ctx.input_end;
    let dummy = DUMMY.as_ptr();

    let mut anchor = istart;
    let mut ip = istart;
    let mut offset_1 = ctx.rep[0];
    let mut offset_2 = ctx.rep[1];
    let mut offset_saved1 = 0u32;
    let mut offset_saved2 = 0u32;

    ip = unsafe { ip.add((ip == prefix_lowest) as usize) };
    {
        let current = unsafe { ip.offset_from(base) } as u32;
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

    macro_rules! hl {
        ($p:expr) => {
            unsafe { hash_ptr::<8>($p, h_bits_l) }
        };
    }
    macro_rules! hs {
        ($p:expr) => {
            unsafe { hash_ptr::<MLS>($p, h_bits_s) }
        };
    }

    'outer: loop {
        let mut step = 1usize;
        let mut next_step = unsafe { ip.add(k_step_incr) };
        let mut ip1 = unsafe { ip.add(step) };
        if ip1 > ilimit {
            break 'outer;
        }
        let mut hl0 = hl!(ip);
        let mut idxl0 = unsafe { *hash_long.add(hl0) };
        let mut matchl0 = unsafe { base.add(idxl0 as usize) };
        let mut curr: u32;

        let mut m_length: usize;
        let mut offset: u32;
        let hl1: usize;
        // Inner loop: one iteration per position searched.
        let stored_rep: bool;
        loop {
            let hs0 = hs!(ip);
            let idxs0 = unsafe { *hash_small.add(hs0) };
            curr = unsafe { ip.offset_from(base) } as u32;
            let matchs0 = unsafe { base.add(idxs0 as usize) };
            unsafe {
                *hash_long.add(hl0) = curr;
                *hash_small.add(hs0) = curr;
            }

            // repcode at ip+1
            if (offset_1 > 0) & unsafe { read32(ip.add(1).sub(offset_1 as usize)) == read32(ip.add(1)) } {
                m_length = unsafe { count(ip.add(5), ip.add(5).sub(offset_1 as usize), iend) } + 4;
                ip = unsafe { ip.add(1) };
                unsafe { seq.store_seq(ip.offset_from(anchor) as usize, anchor, lit_limit, REPCODE1_TO_OFFBASE, m_length) };
                stored_rep = true;
                hl1 = 0;
                offset = 0;
                break;
            }

            let hl1_ = hl!(ip1);
            let matchl0_safe = if idxl0 > prefix_lowest_index { matchl0 } else { dummy };
            if unsafe { read64(matchl0_safe) == read64(ip) } && matchl0_safe == matchl0 {
                m_length = unsafe { count(ip.add(8), matchl0.add(8), iend) } + 8;
                offset = unsafe { ip.offset_from(matchl0) } as u32;
                while ((ip > anchor) & (matchl0 > prefix_lowest)) && unsafe { *ip.sub(1) == *matchl0.sub(1) } {
                    ip = unsafe { ip.sub(1) };
                    matchl0 = unsafe { matchl0.sub(1) };
                    m_length += 1;
                }
                hl1 = hl1_;
                stored_rep = false;
                break;
            }

            let idxl1 = unsafe { *hash_long.add(hl1_) };
            let matchl1 = unsafe { base.add(idxl1 as usize) };
            let matchs0_safe = if idxs0 > prefix_lowest_index { matchs0 } else { dummy };
            if unsafe { read32(matchs0_safe) == read32(ip) } && matchs0_safe == matchs0 {
                // _search_next_long
                let mut ms0 = matchs0;
                m_length = unsafe { count(ip.add(4), ms0.add(4), iend) } + 4;
                offset = unsafe { ip.offset_from(ms0) } as u32;
                if idxl1 > prefix_lowest_index && unsafe { read64(matchl1) == read64(ip1) } {
                    let l1len = unsafe { count(ip1.add(8), matchl1.add(8), iend) } + 8;
                    if l1len > m_length {
                        ip = ip1;
                        m_length = l1len;
                        offset = unsafe { ip.offset_from(matchl1) } as u32;
                        ms0 = matchl1;
                    }
                }
                while ((ip > anchor) & (ms0 > prefix_lowest)) && unsafe { *ip.sub(1) == *ms0.sub(1) } {
                    ip = unsafe { ip.sub(1) };
                    ms0 = unsafe { ms0.sub(1) };
                    m_length += 1;
                }
                hl1 = hl1_;
                stored_rep = false;
                break;
            }

            if ip1 >= next_step {
                step += 1;
                next_step = unsafe { next_step.add(k_step_incr) };
            }
            ip = ip1;
            ip1 = unsafe { ip1.add(step) };
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
                unsafe { *hash_long.add(hl1) = ip1.offset_from(base) as u32 };
            }
            unsafe { seq.store_seq(ip.offset_from(anchor) as usize, anchor, lit_limit, offset_to_offbase(offset), m_length) };
        }

        // _match_stored
        ip = unsafe { ip.add(m_length) };
        anchor = ip;
        if ip <= ilimit {
            unsafe {
                let index_to_insert = curr + 2;
                let p2 = base.add(index_to_insert as usize);
                *hash_long.add(hl!(p2)) = index_to_insert;
                *hash_long.add(hl!(ip.sub(2))) = ip.offset_from(base) as u32 - 2;
                *hash_small.add(hs!(p2)) = index_to_insert;
                *hash_small.add(hs!(ip.sub(1))) = ip.offset_from(base) as u32 - 1;
            }
            while ip <= ilimit && (offset_2 > 0) & unsafe { read32(ip) == read32(ip.sub(offset_2 as usize)) } {
                let r_length = unsafe { count(ip.add(4), ip.add(4).sub(offset_2 as usize), iend) } + 4;
                core::mem::swap(&mut offset_1, &mut offset_2);
                unsafe {
                    *hash_small.add(hs!(ip)) = ip.offset_from(base) as u32;
                    *hash_long.add(hl!(ip)) = ip.offset_from(base) as u32;
                    seq.store_seq(0, anchor, lit_limit, REPCODE1_TO_OFFBASE, r_length);
                }
                ip = unsafe { ip.add(r_length) };
                anchor = ip;
            }
        }
    }

    offset_saved2 = if offset_saved1 != 0 && offset_1 != 0 { offset_saved1 } else { offset_saved2 };
    ctx.rep[0] = if offset_1 != 0 { offset_1 } else { offset_saved1 };
    ctx.rep[1] = if offset_2 != 0 { offset_2 } else { offset_saved2 };
    unsafe { iend.offset_from(anchor) as usize }
}
