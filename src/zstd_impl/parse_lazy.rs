//! Row-hash match finder + greedy/lazy/lazy2 parser — port of
//! `ZSTD_RowFindBestMatch` / `ZSTD_row_*` / `ZSTD_compressBlock_lazy_generic`
//! (C zstd 1.5.7, noDict, search_rowHash). Used for levels 5+.
//!
//! Positions are `usize` indices into `ctx.base`; all byte and table access
//! goes through the unchecked helpers in `seqstore` (see the module docs
//! there). The only other `unsafe` here is the SSE2 tag compare and the
//! prefetch hints.

use super::parse_fast::BlockCtx;
use super::seqstore::{byte, count, hash_at, highbit32, offset_to_offbase, read32, tbl_get, tbl_set, SeqStore, REPCODE1_TO_OFFBASE};

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

    /// Both tables have `1 << (row_hash_log + row_log)` entries, so a row
    /// index (`hash >> TAG_BITS << row_log`, `row_hash_log` bits of hash)
    /// plus any in-row position is in bounds.
    fn check(&self, row_log: u32) {
        let size = 1usize << (self.row_hash_log + row_log);
        assert!(
            self.hash_table.len() == size && self.tag_table.len() == size,
            "zstd row finder: tables not reset for this row_log"
        );
    }
}

/// The row finder's tables and cursors, borrowed from a [`RowState`] for one
/// block. The hot loops work on this stack-local struct (whose fields LLVM
/// promotes to registers once everything is inlined) instead of reaching
/// through `&mut RowState` on every access, which cost ~7% of instructions
/// at level 9 in the index-based port.
struct Tables<'a> {
    hash_table: &'a mut [u32],
    tag_table: &'a mut [u8],
    hash_cache: &'a mut [u32; ROW_HASH_CACHE_SIZE],
    row_hash_log: u32,
    next_to_update: u32,
    lazy_skipping: bool,
}

/// The row `tbl[rel_row .. rel_row + len]`.
///
/// # Safety
/// `rel_row + len <= tbl.len()`: holds for every row index derived from a
/// hash by `RowState::check` (a checked range here cost ~2% of level-9
/// instructions, two per inserted position).
#[inline(always)]
unsafe fn row_mut<T>(tbl: &mut [T], rel_row: usize, len: usize) -> &mut [T] {
    debug_assert!(rel_row + len <= tbl.len());
    // SAFETY: per the contract above.
    unsafe { tbl.get_unchecked_mut(rel_row..rel_row + len) }
}

/// Prefetch hint; never faults, whatever `p` points at.
#[inline(always)]
fn prefetch(p: *const u8) {
    #[cfg(target_arch = "x86_64")]
    // SAFETY: `_mm_prefetch` is a hint and cannot fault; SSE is baseline on
    // x86_64.
    unsafe {
        core::arch::x86_64::_mm_prefetch(p as *const i8, core::arch::x86_64::_MM_HINT_T0);
    }
    #[cfg(not(target_arch = "x86_64"))]
    {
        let _ = p;
    }
}

/// Prefetch position `pos` of `buf` (any `pos`, see [`prefetch`]).
#[inline(always)]
fn prefetch_at(buf: &[u8], pos: usize) {
    prefetch(buf.as_ptr().wrapping_add(pos));
}

/// `ZSTD_row_nextIndex`: advance the row's head (stored in tag slot 0) and
/// return the slot to fill.
///
/// # Safety
/// `rel_row < tag_table.len()`.
#[inline(always)]
unsafe fn row_next_index(tag_table: &mut [u8], rel_row: usize, row_mask: u32) -> u32 {
    // SAFETY: per the contract above.
    let mut next = (unsafe { tbl_get(tag_table, rel_row) } as u32).wrapping_sub(1) & row_mask;
    next += if next == 0 { row_mask } else { 0 };
    // SAFETY: per the contract above.
    unsafe { tbl_set(tag_table, rel_row, next as u8) };
    next
}

#[inline(always)]
fn row_prefetch(hash_table: &[u32], tag_table: &[u8], rel_row: usize, row_log: u32) {
    let h = hash_table.as_ptr().wrapping_add(rel_row) as *const u8;
    prefetch(h);
    if row_log >= 5 {
        prefetch(h.wrapping_add(64));
    }
    prefetch_at(tag_table, rel_row);
    if row_log == 6 {
        prefetch_at(tag_table, rel_row + 32);
    }
}

/// Match mask: bit i set iff `tag_row[i] == tag`, rotated right by `head`.
/// `tag_row` holds the row's `row_entries` (16, 32 or 64) tags.
#[inline(always)]
fn row_get_match_mask(tag_row: &[u8], tag: u8, head: u32, row_entries: u32) -> u64 {
    debug_assert!(tag_row.len() >= row_entries as usize);
    #[cfg(target_arch = "x86_64")]
    // SAFETY: SSE2 is baseline on x86_64, and each 16-byte load is within
    // the first `row_entries <= tag_row.len()` bytes of the slice. The
    // intrinsics are what turn a 16..64-way byte compare into one
    // `pcmpeqb`/`pmovmskb` pair per 16 tags.
    unsafe {
        use core::arch::x86_64::*;
        let p = tag_row.as_ptr();
        let cmp = _mm_set1_epi8(tag as i8);
        let m0 = _mm_movemask_epi8(_mm_cmpeq_epi8(_mm_loadu_si128(p as *const __m128i), cmp)) as u32 as u64;
        if row_entries == 16 {
            return (m0 as u16).rotate_right(head) as u64;
        }
        let m1 = _mm_movemask_epi8(_mm_cmpeq_epi8(_mm_loadu_si128(p.add(16) as *const __m128i), cmp)) as u32 as u64;
        if row_entries == 32 {
            return ((m1 << 16 | m0) as u32).rotate_right(head) as u64;
        }
        let m2 = _mm_movemask_epi8(_mm_cmpeq_epi8(_mm_loadu_si128(p.add(32) as *const __m128i), cmp)) as u32 as u64;
        let m3 = _mm_movemask_epi8(_mm_cmpeq_epi8(_mm_loadu_si128(p.add(48) as *const __m128i), cmp)) as u32 as u64;
        (m3 << 48 | m2 << 32 | m1 << 16 | m0).rotate_right(head)
    }
    #[cfg(not(target_arch = "x86_64"))]
    {
        let mut matches: u64 = 0;
        for &t in tag_row[..row_entries as usize].iter().rev() {
            matches = (matches << 1) | (t == tag) as u64;
        }
        match row_entries {
            16 => (matches as u16).rotate_right(head) as u64,
            32 => (matches as u32).rotate_right(head) as u64,
            _ => matches.rotate_right(head),
        }
    }
}

/// `ZSTD_row_fillHashCache`: hash positions `idx ..= i_limit` (at most
/// `ROW_HASH_CACHE_SIZE` of them) into the cache and prefetch their rows.
///
/// # Safety
/// `i_limit + 8 <= base.len()`; tables checked for `row_log`.
#[inline(always)]
unsafe fn row_fill_hash_cache<const MLS: u32>(t: &mut Tables, base: &[u8], row_log: u32, mut idx: u32, i_limit: usize) {
    let hash_log = t.row_hash_log;
    let max_elems = if idx as usize > i_limit { 0 } else { (i_limit - idx as usize) as u32 + 1 };
    let lim = idx + (ROW_HASH_CACHE_SIZE as u32).min(max_elems);
    while idx < lim {
        // SAFETY: idx <= i_limit, so 8 bytes are readable.
        let hash = unsafe { hash_at::<MLS>(base, idx as usize, hash_log + ROW_HASH_TAG_BITS) } as u32;
        let row = ((hash >> ROW_HASH_TAG_BITS) << row_log) as usize;
        row_prefetch(t.hash_table, t.tag_table, row, row_log);
        t.hash_cache[idx as usize & ROW_HASH_CACHE_MASK] = hash;
        idx += 1;
    }
}

/// `ZSTD_row_nextCachedHash`: take position `idx`'s hash from the cache and
/// refill the slot with the hash of `idx + ROW_HASH_CACHE_SIZE`.
///
/// # Safety
/// `idx + ROW_HASH_CACHE_SIZE + 8 <= base.len()`.
#[inline(always)]
unsafe fn row_next_cached_hash<const MLS: u32>(t: &mut Tables, base: &[u8], idx: u32, row_log: u32) -> u32 {
    // SAFETY: per the contract above.
    let new_hash = unsafe { hash_at::<MLS>(base, idx as usize + ROW_HASH_CACHE_SIZE, t.row_hash_log + ROW_HASH_TAG_BITS) } as u32;
    let row = ((new_hash >> ROW_HASH_TAG_BITS) << row_log) as usize;
    row_prefetch(t.hash_table, t.tag_table, row, row_log);
    let slot = idx as usize & ROW_HASH_CACHE_MASK;
    let hash = t.hash_cache[slot];
    t.hash_cache[slot] = new_hash;
    hash
}

/// `ZSTD_row_update_internalImpl`: insert positions `start .. end`.
///
/// # Safety
/// `end + ROW_HASH_CACHE_SIZE + 8 <= base.len()`; tables checked for
/// `row_log`.
#[inline(always)]
unsafe fn row_update_internal_impl<const MLS: u32>(t: &mut Tables, mut start: u32, end: u32, base: &[u8], row_log: u32, row_mask: u32, use_cache: bool) {
    let hash_log = t.row_hash_log;
    while start < end {
        let hash = if use_cache {
            // SAFETY: start < end, per the contract above.
            unsafe { row_next_cached_hash::<MLS>(t, base, start, row_log) }
        } else {
            // SAFETY: start < end, per the contract above.
            unsafe { hash_at::<MLS>(base, start as usize, hash_log + ROW_HASH_TAG_BITS) as u32 }
        };
        let rel_row = ((hash >> ROW_HASH_TAG_BITS) << row_log) as usize;
        let row_len = row_mask as usize + 1;
        // SAFETY: rows are in bounds by `RowState::check`; `pos <= row_mask
        // < row_len` and slot 0 exists.
        unsafe {
            let row = row_mut(t.hash_table, rel_row, row_len);
            let tag_row = row_mut(t.tag_table, rel_row, row_len);
            let pos = row_next_index(tag_row, 0, row_mask) as usize;
            tbl_set(tag_row, pos, (hash & ROW_HASH_TAG_MASK) as u8);
            tbl_set(row, pos, start);
        }
        start += 1;
    }
}

/// `ZSTD_row_update_internal`: catch the tables up to `ip`.
///
/// # Safety
/// `ip + ROW_HASH_CACHE_SIZE + 8 <= base.len()`; tables checked for
/// `row_log`.
#[inline(always)]
unsafe fn row_update_internal<const MLS: u32>(t: &mut Tables, ip: usize, base: &[u8], row_log: u32, row_mask: u32) {
    let mut idx = t.next_to_update;
    let target = ip as u32;
    const K_SKIP_THRESHOLD: u32 = 384;
    const K_MAX_MATCH_START_POSITIONS_TO_UPDATE: u32 = 96;
    const K_MAX_MATCH_END_POSITIONS_TO_UPDATE: u32 = 32;
    // SAFETY: every position touched is <= ip + 1, per the contract above.
    unsafe {
        if target - idx > K_SKIP_THRESHOLD {
            let bound = idx + K_MAX_MATCH_START_POSITIONS_TO_UPDATE;
            row_update_internal_impl::<MLS>(t, idx, bound, base, row_log, row_mask, true);
            idx = target - K_MAX_MATCH_END_POSITIONS_TO_UPDATE;
            row_fill_hash_cache::<MLS>(t, base, row_log, idx, ip + 1);
        }
        row_update_internal_impl::<MLS>(t, idx, target, base, row_log, row_mask, true);
    }
    t.next_to_update = target;
}

/// `ZSTD_RowFindBestMatch` (noDict). Returns the match length (3 if none)
/// and writes `off_base` on success.
///
/// # Safety
/// `ip + ROW_HASH_CACHE_SIZE + 8 <= i_limit <= base.len()`; tables checked
/// for `row_log` and holding only positions `< ip` (stored by this finder
/// for a prefix of `base`).
#[inline(always)]
#[allow(clippy::too_many_arguments)]
unsafe fn row_find_best_match<const MLS: u32>(
    t: &mut Tables,
    ip: usize,
    i_limit: usize,
    off_base: &mut u32,
    base: &[u8],
    window_log: u32,
    search_log: u32,
    row_log: u32,
) -> usize {
    let curr = ip as u32;
    let max_distance = 1u32 << window_log;
    let low_limit = if curr > max_distance { curr - max_distance } else { 0 };
    let row_entries = 1u32 << row_log;
    let row_mask = row_entries - 1;
    let capped_search_log = search_log.min(row_log);
    let mut nb_attempts = 1u32 << capped_search_log;
    let mut ml: usize = 4 - 1;

    let hash = if !t.lazy_skipping {
        // SAFETY: per the contract above.
        unsafe {
            row_update_internal::<MLS>(t, ip, base, row_log, row_mask);
            row_next_cached_hash::<MLS>(t, base, curr, row_log)
        }
    } else {
        t.next_to_update = curr;
        // SAFETY: per the contract above.
        unsafe { hash_at::<MLS>(base, ip, t.row_hash_log + ROW_HASH_TAG_BITS) as u32 }
    };

    let rel_row = ((hash >> ROW_HASH_TAG_BITS) << row_log) as usize;
    let tag = (hash & ROW_HASH_TAG_MASK) as u8;
    let row_len = row_entries as usize;
    // SAFETY: rows are in bounds by `RowState::check`.
    let (row, tag_row) = unsafe { (&*row_mut(t.hash_table, rel_row, row_len), &*row_mut(t.tag_table, rel_row, row_len)) };
    let head = tag_row[0] as u32 & row_mask;
    let mut match_buffer = [0u32; ROW_HASH_MAX_ENTRIES];
    let mut num_matches = 0usize;
    let mut matches = row_get_match_mask(tag_row, tag, head, row_entries);

    while matches > 0 && nb_attempts > 0 {
        let match_pos = (head + matches.trailing_zeros()) & row_mask;
        // SAFETY: `match_pos <= row_mask < row.len()`.
        let match_index = unsafe { tbl_get(row, match_pos as usize) };
        matches &= matches - 1;
        if match_pos == 0 {
            continue;
        }
        if match_index < low_limit {
            break;
        }
        prefetch_at(base, match_index as usize);
        match_buffer[num_matches] = match_index;
        num_matches += 1;
        nb_attempts -= 1;
    }

    // Insert the current position too.
    // SAFETY: rows are in bounds by `RowState::check`; `pos <= row_mask <
    // row_len` and slot 0 exists (`row_len >= 16`).
    unsafe {
        let row = row_mut(t.hash_table, rel_row, row_len);
        let tag_row = row_mut(t.tag_table, rel_row, row_len);
        let pos = row_next_index(tag_row, 0, row_mask) as usize;
        tbl_set(tag_row, pos, tag);
        tbl_set(row, pos, t.next_to_update);
    }
    t.next_to_update += 1;

    for &match_index in &match_buffer[..num_matches] {
        let m = match_index as usize;
        // SAFETY: `m < ip` (table contents, contract) and `ip + ml + 1 <=
        // i_limit`: `ml` starts at 3 with `ip + 4 <= i_limit`, and the loop
        // exits below as soon as `ip + ml == i_limit`.
        let current_ml = if unsafe { read32(base, m + ml - 3) == read32(base, ip + ml - 3) } {
            // SAFETY: m < ip <= i_limit <= base.len().
            unsafe { count(base, ip, m, i_limit) }
        } else {
            0
        };
        if current_ml > ml {
            ml = current_ml;
            *off_base = offset_to_offbase(curr - match_index);
            if ip + current_ml == i_limit {
                break;
            }
        }
    }
    ml
}

/// `ZSTD_compressBlock_lazy_generic` (noDict, rowHash). `depth`: 0 greedy,
/// 1 lazy, 2 lazy2. Returns the number of trailing literals.
///
/// # Safety
/// `st` was `reset` for this `search_log` (table sizes are asserted) and
/// since then has only been fed prefixes of `ctx.base` by this function, so
/// every stored position is `< ctx.base.len()`. `seq.reset(ctx.block_len)`
/// was called for this block.
pub unsafe fn compress_block_lazy(st: &mut RowState, seq: &mut SeqStore, ctx: &mut BlockCtx, search_log: u32, mls: u32, depth: u32) -> usize {
    ctx.check();
    st.check(search_log.clamp(4, 6));
    let mls = mls.clamp(4, 6);
    // SAFETY: the caller's contract, plus the checks above.
    unsafe {
        match mls {
            4 => lazy_generic::<4>(st, seq, ctx, search_log, depth),
            5 => lazy_generic::<5>(st, seq, ctx, search_log, depth),
            _ => lazy_generic::<6>(st, seq, ctx, search_log, depth),
        }
    }
}

/// # Safety
/// As [`compress_block_lazy`].
#[inline(never)]
unsafe fn lazy_generic<const MLS: u32>(st: &mut RowState, seq: &mut SeqStore, ctx: &mut BlockCtx, search_log: u32, depth: u32) -> usize {
    let base = ctx.base;
    let istart = ctx.istart as usize;
    let iend = istart + ctx.block_len;
    // C: `iend - (8 + ROW_HASH_CACHE_SIZE)`; saturating keeps the
    // `ip < ilimit` exits equivalent for blocks too short to search.
    let ilimit = iend.saturating_sub(8 + ROW_HASH_CACHE_SIZE);
    let prefix_lowest = 0usize;
    let row_log = search_log.clamp(4, 6);
    let window_log = ctx.window_log;

    let mut anchor = istart;
    let mut ip = istart;
    let mut offset_1 = ctx.rep[0];
    let mut offset_2 = ctx.rep[1];
    let mut offset_saved1 = 0u32;
    let mut offset_saved2 = 0u32;

    ip += (ip == prefix_lowest) as usize;
    {
        let curr = ip as u32;
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
    let mut t = Tables {
        hash_table: &mut st.hash_table,
        tag_table: &mut st.tag_table,
        hash_cache: &mut st.hash_cache,
        row_hash_log: st.row_hash_log,
        next_to_update: st.next_to_update,
        lazy_skipping: false,
    };
    let t = &mut t;
    let ntu = t.next_to_update;

    // SAFETY invariants for everything below (`iend <= base.len()` by
    // `check`, tables by `RowState::check`):
    //  (a) `ip < ilimit = iend - 16` wherever bytes are read at/after `ip`
    //      (the repcode checks read 4 bytes at `ip`/`ip + 1` and `count`
    //      from `ip + 4`/`ip + 5`); searches need `ip + 16 <= iend`.
    //  (b) offset_1/2 <= the current position: validated against `max_rep`
    //      above, afterwards only set to `curr - match_index` for a found
    //      match (`match_index >= low_limit >= 0`); reads at `ip - offset`
    //      are masked by `offset > 0`.
    //  (c) the hash cache and table updates only touch positions
    //      `<= ip + 1 + ROW_HASH_CACHE_SIZE` with `ip < ilimit`, or
    //      `<= ilimit` for `row_fill_hash_cache`, so 8 bytes are readable.
    //  (d) `count` is called with `match < ip <= iend`.
    // SAFETY: (c)
    unsafe { row_fill_hash_cache::<MLS>(t, base, row_log, ntu, ilimit) };

    macro_rules! search {
        ($ip:expr, $ob:expr) => {
            // SAFETY: (a), (c)
            unsafe { row_find_best_match::<MLS>(t, $ip, iend, $ob, base, window_log, search_log, row_log) }
        };
    }

    while ip < ilimit {
        let mut match_length: usize = 0;
        let mut off_base: u32 = REPCODE1_TO_OFFBASE;
        let mut start = ip + 1;

        // repcode at ip+1
        let mut go_store = false;
        // SAFETY: (a), (b)
        if (offset_1 > 0) & unsafe { read32(base, ip + 1 - offset_1 as usize) == read32(base, ip + 1) } {
            // SAFETY: (d)
            match_length = unsafe { count(base, ip + 5, ip + 5 - offset_1 as usize, iend) } + 4;
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
                let step = ((ip - anchor) >> K_SEARCH_STRENGTH) + 1;
                ip += step;
                t.lazy_skipping = step > K_LAZY_SKIPPING_STEP;
                continue;
            }

            if depth >= 1 {
                while ip < ilimit {
                    ip += 1;
                    // SAFETY: (a), (b)
                    if (offset_1 > 0) & unsafe { read32(base, ip) == read32(base, ip - offset_1 as usize) } {
                        // SAFETY: (d)
                        let ml_rep = unsafe { count(base, ip + 4, ip + 4 - offset_1 as usize, iend) } + 4;
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
                        ip += 1;
                        // SAFETY: (a), (b)
                        if (offset_1 > 0) & unsafe { read32(base, ip) == read32(base, ip - offset_1 as usize) } {
                            // SAFETY: (d)
                            let ml_rep = unsafe { count(base, ip + 4, ip + 4 - offset_1 as usize, iend) } + 4;
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
                // SAFETY: `start > anchor >= 0` and `start - offset >
                // prefix_lowest = 0` keep both `- 1` positions in range.
                while ((start > anchor) & (start - offset > prefix_lowest)) && unsafe { byte(base, start - 1) == byte(base, start - offset - 1) } {
                    start -= 1;
                    match_length += 1;
                }
                offset_2 = offset_1;
                offset_1 = offset as u32;
            }
        }

        // _storeSequence
        // SAFETY: the literals lie in `base` (anchor <= position <= iend <= base.len()) and `seq.reset(block_len)` covers this block.
        unsafe { seq.store_seq(anchor, start - anchor, base, off_base, match_length) };
        ip = start + match_length;
        anchor = ip;

        if t.lazy_skipping {
            let ntu = t.next_to_update;
            // SAFETY: (c)
            unsafe { row_fill_hash_cache::<MLS>(t, base, row_log, ntu, ilimit) };
            t.lazy_skipping = false;
        }

        // SAFETY: (a) (`ip <= ilimit` here), (b)
        while ((ip <= ilimit) & (offset_2 > 0)) && unsafe { read32(base, ip) == read32(base, ip - offset_2 as usize) } {
            // SAFETY: (d)
            let ml = unsafe { count(base, ip + 4, ip + 4 - offset_2 as usize, iend) } + 4;
            core::mem::swap(&mut offset_1, &mut offset_2);
            // SAFETY: the literals lie in `base` (anchor <= position <= iend <= base.len()) and `seq.reset(block_len)` covers this block.
            unsafe { seq.store_seq(anchor, 0, base, REPCODE1_TO_OFFBASE, ml) };
            ip += ml;
            anchor = ip;
        }
    }

    st.next_to_update = t.next_to_update;
    st.lazy_skipping = t.lazy_skipping;
    offset_saved2 = if offset_saved1 != 0 && offset_1 != 0 { offset_saved1 } else { offset_saved2 };
    ctx.rep[0] = if offset_1 != 0 { offset_1 } else { offset_saved1 };
    ctx.rep[1] = if offset_2 != 0 { offset_2 } else { offset_saved2 };
    iend - anchor
}
