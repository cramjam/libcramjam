//! Row-hash match finder + greedy/lazy/lazy2 parser — port of
//! `ZSTD_RowFindBestMatch` / `ZSTD_row_*` / `ZSTD_compressBlock_lazy_generic`
//! (C zstd 1.5.7, noDict, search_rowHash). Used for levels 5+.
#![allow(unused_unsafe)]

use super::parse_fast::BlockCtx;
use super::seqstore::{count, hash_ptr, highbit32, offset_to_offbase, read32, SeqStore, REPCODE1_TO_OFFBASE};

const ROW_HASH_TAG_BITS: u32 = 8;
const ROW_HASH_TAG_MASK: u32 = (1 << ROW_HASH_TAG_BITS) - 1;
const ROW_HASH_CACHE_SIZE: usize = 8;
const ROW_HASH_CACHE_MASK: usize = ROW_HASH_CACHE_SIZE - 1;
const ROW_HASH_MAX_ENTRIES: usize = 64;
const K_SEARCH_STRENGTH: u32 = 8;
const K_LAZY_SKIPPING_STEP: usize = 8;

pub struct RowState {
    pub hash_table: Vec<u32>,
    pub tag_table: Vec<u8>,
    hash_cache: [u32; ROW_HASH_CACHE_SIZE],
    pub next_to_update: u32,
    row_hash_log: u32,
    lazy_skipping: bool,
}

impl RowState {
    pub fn new() -> Self {
        RowState {
            hash_table: Vec::new(),
            tag_table: Vec::new(),
            hash_cache: [0; ROW_HASH_CACHE_SIZE],
            next_to_update: 0,
            row_hash_log: 0,
            lazy_skipping: false,
        }
    }

    pub fn reset(&mut self, hash_log: u32, row_log: u32) {
        let size = 1usize << hash_log;
        self.hash_table.clear();
        self.hash_table.resize(size, 0);
        self.tag_table.clear();
        self.tag_table.resize(size, 0);
        self.hash_cache = [0; ROW_HASH_CACHE_SIZE];
        self.next_to_update = 0;
        self.row_hash_log = hash_log - row_log;
        self.lazy_skipping = false;
    }
}

#[inline(always)]
fn prefetch(p: *const u8) {
    #[cfg(target_arch = "x86_64")]
    unsafe {
        core::arch::x86_64::_mm_prefetch(p as *const i8, core::arch::x86_64::_MM_HINT_T0);
    }
    #[cfg(not(target_arch = "x86_64"))]
    {
        let _ = p;
    }
}

#[inline(always)]
unsafe fn row_next_index(tag_row: *mut u8, row_mask: u32) -> u32 {
    let mut next = (unsafe { *tag_row } as u32).wrapping_sub(1) & row_mask;
    next += if next == 0 { row_mask } else { 0 };
    unsafe { *tag_row = next as u8 };
    next
}

#[inline(always)]
fn row_prefetch(hash_table: *const u32, tag_table: *const u8, rel_row: usize, row_log: u32) {
    unsafe {
        prefetch(hash_table.add(rel_row) as *const u8);
        if row_log >= 5 {
            prefetch(hash_table.add(rel_row + 16) as *const u8);
        }
        prefetch(tag_table.add(rel_row));
        if row_log == 6 {
            prefetch(tag_table.add(rel_row + 32));
        }
    }
}

/// Match mask: bit i set iff `tag_row[i] == tag`, rotated right by `head`.
#[inline(always)]
unsafe fn row_get_match_mask(tag_row: *const u8, tag: u8, head: u32, row_entries: u32) -> u64 {
    #[cfg(target_arch = "x86_64")]
    unsafe {
        use core::arch::x86_64::*;
        let cmp = _mm_set1_epi8(tag as i8);
        let m0 = _mm_movemask_epi8(_mm_cmpeq_epi8(_mm_loadu_si128(tag_row as *const __m128i), cmp)) as u32 as u64;
        if row_entries == 16 {
            return (m0 as u16).rotate_right(head) as u64;
        }
        let m1 = _mm_movemask_epi8(_mm_cmpeq_epi8(_mm_loadu_si128(tag_row.add(16) as *const __m128i), cmp)) as u32 as u64;
        if row_entries == 32 {
            return ((m1 << 16 | m0) as u32).rotate_right(head) as u64;
        }
        let m2 = _mm_movemask_epi8(_mm_cmpeq_epi8(_mm_loadu_si128(tag_row.add(32) as *const __m128i), cmp)) as u32 as u64;
        let m3 = _mm_movemask_epi8(_mm_cmpeq_epi8(_mm_loadu_si128(tag_row.add(48) as *const __m128i), cmp)) as u32 as u64;
        (m3 << 48 | m2 << 32 | m1 << 16 | m0).rotate_right(head)
    }
    #[cfg(not(target_arch = "x86_64"))]
    unsafe {
        let mut matches: u64 = 0;
        for i in (0..row_entries as usize).rev() {
            matches = (matches << 1) | (*tag_row.add(i) == tag) as u64;
        }
        match row_entries {
            16 => (matches as u16).rotate_right(head) as u64,
            32 => (matches as u32).rotate_right(head) as u64,
            _ => matches.rotate_right(head),
        }
    }
}

#[inline(always)]
unsafe fn row_fill_hash_cache<const MLS: u32>(st: &mut RowState, base: *const u8, row_log: u32, mut idx: u32, i_limit: *const u8) {
    let hash_log = st.row_hash_log;
    let p = unsafe { base.add(idx as usize) };
    let max_elems = if p > i_limit { 0 } else { (unsafe { i_limit.offset_from(p) }) as u32 + 1 };
    let lim = idx + (ROW_HASH_CACHE_SIZE as u32).min(max_elems);
    while idx < lim {
        let hash = unsafe { hash_ptr::<MLS>(base.add(idx as usize), hash_log + ROW_HASH_TAG_BITS) } as u32;
        let row = ((hash >> ROW_HASH_TAG_BITS) << row_log) as usize;
        row_prefetch(st.hash_table.as_ptr(), st.tag_table.as_ptr(), row, row_log);
        st.hash_cache[idx as usize & ROW_HASH_CACHE_MASK] = hash;
        idx += 1;
    }
}

#[inline(always)]
unsafe fn row_next_cached_hash<const MLS: u32>(st: &mut RowState, base: *const u8, idx: u32, row_log: u32) -> u32 {
    let new_hash = unsafe { hash_ptr::<MLS>(base.add(idx as usize + ROW_HASH_CACHE_SIZE), st.row_hash_log + ROW_HASH_TAG_BITS) } as u32;
    let row = ((new_hash >> ROW_HASH_TAG_BITS) << row_log) as usize;
    row_prefetch(st.hash_table.as_ptr(), st.tag_table.as_ptr(), row, row_log);
    let slot = idx as usize & ROW_HASH_CACHE_MASK;
    let hash = st.hash_cache[slot];
    st.hash_cache[slot] = new_hash;
    hash
}

#[inline(always)]
unsafe fn row_update_internal_impl<const MLS: u32>(st: &mut RowState, mut start: u32, end: u32, base: *const u8, row_log: u32, row_mask: u32, use_cache: bool) {
    let hash_log = st.row_hash_log;
    let hash_table = st.hash_table.as_mut_ptr();
    let tag_table = st.tag_table.as_mut_ptr();
    while start < end {
        let hash = if use_cache {
            unsafe { row_next_cached_hash::<MLS>(st, base, start, row_log) }
        } else {
            unsafe { hash_ptr::<MLS>(base.add(start as usize), hash_log + ROW_HASH_TAG_BITS) as u32 }
        };
        let rel_row = ((hash >> ROW_HASH_TAG_BITS) << row_log) as usize;
        unsafe {
            let row = hash_table.add(rel_row);
            let tag_row = tag_table.add(rel_row);
            let pos = row_next_index(tag_row, row_mask);
            *tag_row.add(pos as usize) = (hash & ROW_HASH_TAG_MASK) as u8;
            *row.add(pos as usize) = start;
        }
        start += 1;
    }
}

#[inline(always)]
unsafe fn row_update_internal<const MLS: u32>(st: &mut RowState, ip: *const u8, base: *const u8, row_log: u32, row_mask: u32) {
    let mut idx = st.next_to_update;
    let target = unsafe { ip.offset_from(base) } as u32;
    const K_SKIP_THRESHOLD: u32 = 384;
    const K_MAX_MATCH_START_POSITIONS_TO_UPDATE: u32 = 96;
    const K_MAX_MATCH_END_POSITIONS_TO_UPDATE: u32 = 32;
    if target - idx > K_SKIP_THRESHOLD {
        let bound = idx + K_MAX_MATCH_START_POSITIONS_TO_UPDATE;
        unsafe { row_update_internal_impl::<MLS>(st, idx, bound, base, row_log, row_mask, true) };
        idx = target - K_MAX_MATCH_END_POSITIONS_TO_UPDATE;
        unsafe { row_fill_hash_cache::<MLS>(st, base, row_log, idx, ip.add(1)) };
    }
    unsafe { row_update_internal_impl::<MLS>(st, idx, target, base, row_log, row_mask, true) };
    st.next_to_update = target;
}

/// `ZSTD_RowFindBestMatch` (noDict). Returns the match length (3 if none)
/// and writes `off_base` on success.
#[inline(always)]
unsafe fn row_find_best_match<const MLS: u32>(
    st: &mut RowState,
    ip: *const u8,
    i_limit: *const u8,
    off_base: &mut u32,
    base: *const u8,
    window_log: u32,
    search_log: u32,
    row_log: u32,
) -> usize {
    let curr = unsafe { ip.offset_from(base) } as u32;
    let max_distance = 1u32 << window_log;
    let low_limit = if curr > max_distance { curr - max_distance } else { 0 };
    let row_entries = 1u32 << row_log;
    let row_mask = row_entries - 1;
    let capped_search_log = search_log.min(row_log);
    let mut nb_attempts = 1u32 << capped_search_log;
    let mut ml: usize = 4 - 1;

    let hash = if !st.lazy_skipping {
        unsafe { row_update_internal::<MLS>(st, ip, base, row_log, row_mask) };
        unsafe { row_next_cached_hash::<MLS>(st, base, curr, row_log) }
    } else {
        st.next_to_update = curr;
        unsafe { hash_ptr::<MLS>(ip, st.row_hash_log + ROW_HASH_TAG_BITS) as u32 }
    };

    let rel_row = ((hash >> ROW_HASH_TAG_BITS) << row_log) as usize;
    let tag = (hash & ROW_HASH_TAG_MASK) as u8;
    let row = unsafe { st.hash_table.as_mut_ptr().add(rel_row) };
    let tag_row = unsafe { st.tag_table.as_mut_ptr().add(rel_row) };
    let head = unsafe { *tag_row } as u32 & row_mask;
    let mut match_buffer = [0u32; ROW_HASH_MAX_ENTRIES];
    let mut num_matches = 0usize;
    let mut matches = unsafe { row_get_match_mask(tag_row, tag, head, row_entries) };

    while matches > 0 && nb_attempts > 0 {
        let match_pos = (head + matches.trailing_zeros()) & row_mask;
        let match_index = unsafe { *row.add(match_pos as usize) };
        matches &= matches - 1;
        if match_pos == 0 {
            continue;
        }
        if match_index < low_limit {
            break;
        }
        prefetch(unsafe { base.add(match_index as usize) });
        match_buffer[num_matches] = match_index;
        num_matches += 1;
        nb_attempts -= 1;
    }

    // Insert the current position too.
    unsafe {
        let pos = row_next_index(tag_row, row_mask);
        *tag_row.add(pos as usize) = tag;
        *row.add(pos as usize) = st.next_to_update;
        st.next_to_update += 1;
    }

    for &match_index in &match_buffer[..num_matches] {
        let m = unsafe { base.add(match_index as usize) };
        let current_ml = if unsafe { read32(m.add(ml - 3)) == read32(ip.add(ml - 3)) } {
            unsafe { count(ip, m, i_limit) }
        } else {
            0
        };
        if current_ml > ml {
            ml = current_ml;
            *off_base = offset_to_offbase(curr - match_index);
            if unsafe { ip.add(current_ml) } == i_limit {
                break;
            }
        }
    }
    ml
}

/// `ZSTD_compressBlock_lazy_generic` (noDict, rowHash). `depth`: 0 greedy,
/// 1 lazy, 2 lazy2. Returns the number of trailing literals.
pub unsafe fn compress_block_lazy(st: &mut RowState, seq: &mut SeqStore, ctx: &mut BlockCtx, search_log: u32, mls: u32, depth: u32) -> usize {
    let mls = mls.clamp(4, 6);
    match mls {
        4 => unsafe { lazy_generic::<4>(st, seq, ctx, search_log, depth) },
        5 => unsafe { lazy_generic::<5>(st, seq, ctx, search_log, depth) },
        _ => unsafe { lazy_generic::<6>(st, seq, ctx, search_log, depth) },
    }
}

#[inline(never)]
unsafe fn lazy_generic<const MLS: u32>(st: &mut RowState, seq: &mut SeqStore, ctx: &mut BlockCtx, search_log: u32, depth: u32) -> usize {
    let base = ctx.base;
    let istart = unsafe { base.add(ctx.istart as usize) };
    let iend = unsafe { istart.add(ctx.block_len) };
    let ilimit = unsafe { iend.sub(8 + ROW_HASH_CACHE_SIZE) };
    let prefix_lowest = base;
    let row_log = search_log.clamp(4, 6);
    let window_log = ctx.window_log;
    let lit_limit = ctx.input_end;

    let mut anchor = istart;
    let mut ip = istart;
    let mut offset_1 = ctx.rep[0];
    let mut offset_2 = ctx.rep[1];
    let mut offset_saved1 = 0u32;
    let mut offset_saved2 = 0u32;

    ip = unsafe { ip.add((ip == prefix_lowest) as usize) };
    {
        let curr = unsafe { ip.offset_from(base) } as u32;
        let max_distance = 1u32 << window_log;
        let window_low = if curr > max_distance { curr - max_distance } else { 0 };
        let max_rep = curr - window_low;
        if offset_2 > max_rep {
            offset_saved2 = offset_2;
            offset_2 = 0;
        }
        if offset_1 > max_rep {
            offset_saved1 = offset_1;
            offset_1 = 0;
        }
    }
    st.lazy_skipping = false;
    let ntu = st.next_to_update;
    unsafe { row_fill_hash_cache::<MLS>(st, base, row_log, ntu, ilimit) };

    macro_rules! search {
        ($ip:expr, $ob:expr) => {
            unsafe { row_find_best_match::<MLS>(st, $ip, iend, $ob, base, window_log, search_log, row_log) }
        };
    }

    while ip < ilimit {
        let mut match_length: usize = 0;
        let mut off_base: u32 = REPCODE1_TO_OFFBASE;
        let mut start = unsafe { ip.add(1) };

        // repcode at ip+1
        let mut go_store = false;
        if (offset_1 > 0) & unsafe { read32(ip.add(1).sub(offset_1 as usize)) == read32(ip.add(1)) } {
            match_length = unsafe { count(ip.add(5), ip.add(5).sub(offset_1 as usize), iend) } + 4;
            if depth == 0 {
                go_store = true;
            }
        }
        if !go_store {
            let mut offbase_found: u32 = 999_999_999;
            let ml2 = search!(ip, &mut offbase_found);
            if ml2 > match_length {
                match_length = ml2;
                start = ip;
                off_base = offbase_found;
            }
            if match_length < 4 {
                let step = (unsafe { ip.offset_from(anchor) } as usize >> K_SEARCH_STRENGTH) + 1;
                ip = unsafe { ip.add(step) };
                st.lazy_skipping = step > K_LAZY_SKIPPING_STEP;
                continue;
            }

            if depth >= 1 {
                while ip < ilimit {
                    ip = unsafe { ip.add(1) };
                    if (offset_1 > 0) & unsafe { read32(ip) == read32(ip.sub(offset_1 as usize)) } {
                        let ml_rep = unsafe { count(ip.add(4), ip.add(4).sub(offset_1 as usize), iend) } + 4;
                        let gain2 = (ml_rep * 3) as i32;
                        let gain1 = (match_length * 3) as i32 - highbit32(off_base) as i32 + 1;
                        if ml_rep >= 4 && gain2 > gain1 {
                            match_length = ml_rep;
                            off_base = REPCODE1_TO_OFFBASE;
                            start = ip;
                        }
                    }
                    {
                        let mut ofb_candidate: u32 = 999_999_999;
                        let ml2 = search!(ip, &mut ofb_candidate);
                        let gain2 = (ml2 * 4) as i32 - highbit32(ofb_candidate) as i32;
                        let gain1 = (match_length * 4) as i32 - highbit32(off_base) as i32 + 4;
                        if ml2 >= 4 && gain2 > gain1 {
                            match_length = ml2;
                            off_base = ofb_candidate;
                            start = ip;
                            continue;
                        }
                    }
                    if depth == 2 && ip < ilimit {
                        ip = unsafe { ip.add(1) };
                        if (offset_1 > 0) & unsafe { read32(ip) == read32(ip.sub(offset_1 as usize)) } {
                            let ml_rep = unsafe { count(ip.add(4), ip.add(4).sub(offset_1 as usize), iend) } + 4;
                            let gain2 = (ml_rep * 4) as i32;
                            let gain1 = (match_length * 4) as i32 - highbit32(off_base) as i32 + 1;
                            if ml_rep >= 4 && gain2 > gain1 {
                                match_length = ml_rep;
                                off_base = REPCODE1_TO_OFFBASE;
                                start = ip;
                            }
                        }
                        {
                            let mut ofb_candidate: u32 = 999_999_999;
                            let ml2 = search!(ip, &mut ofb_candidate);
                            let gain2 = (ml2 * 4) as i32 - highbit32(ofb_candidate) as i32;
                            let gain1 = (match_length * 4) as i32 - highbit32(off_base) as i32 + 7;
                            if ml2 >= 4 && gain2 > gain1 {
                                match_length = ml2;
                                off_base = ofb_candidate;
                                start = ip;
                                continue;
                            }
                        }
                    }
                    break;
                }
            }

            // catch up
            if off_base > 3 {
                let offset = (off_base - 3) as usize;
                while ((start > anchor) & (unsafe { start.sub(offset) } > prefix_lowest)) && unsafe { *start.sub(1) == *start.sub(offset).sub(1) } {
                    start = unsafe { start.sub(1) };
                    match_length += 1;
                }
                offset_2 = offset_1;
                offset_1 = offset as u32;
            }
        }

        // _storeSequence
        let lit_length = unsafe { start.offset_from(anchor) } as usize;
        unsafe { seq.store_seq(lit_length, anchor, lit_limit, off_base, match_length) };
        ip = unsafe { start.add(match_length) };
        anchor = ip;

        if st.lazy_skipping {
            let ntu = st.next_to_update;
            unsafe { row_fill_hash_cache::<MLS>(st, base, row_log, ntu, ilimit) };
            st.lazy_skipping = false;
        }

        while ((ip <= ilimit) & (offset_2 > 0)) && unsafe { read32(ip) == read32(ip.sub(offset_2 as usize)) } {
            let ml = unsafe { count(ip.add(4), ip.add(4).sub(offset_2 as usize), iend) } + 4;
            core::mem::swap(&mut offset_1, &mut offset_2);
            unsafe { seq.store_seq(0, anchor, lit_limit, REPCODE1_TO_OFFBASE, ml) };
            ip = unsafe { ip.add(ml) };
            anchor = ip;
        }
    }

    offset_saved2 = if offset_saved1 != 0 && offset_1 != 0 { offset_saved1 } else { offset_saved2 };
    ctx.rep[0] = if offset_1 != 0 { offset_1 } else { offset_saved1 };
    ctx.rep[1] = if offset_2 != 0 { offset_2 } else { offset_saved2 };
    unsafe { iend.offset_from(anchor) as usize }
}
