//! LZ4 HC block compressor — a faithful port of `lz4hc.c` (lz4 1.10):
//!
//! * `LZ4HC_Insert` / `LZ4HC_InsertAndGetWiderMatch`: 32 K-entry hash head
//!   table + 64 K ring of `u16` chain deltas, lazy insertion up to `ip`,
//!   candidate pre-check on the two bytes that would make the match longer,
//!   backward extension (`countBack`), repeated-pattern analysis for levels
//!   9+ (`countPattern` / `reverseCountPattern`) and the chain-swap
//!   heuristic used by the optimal parser.
//! * `LZ4HC_compress_hashChain` (levels 3-9): the `_Search2` / `_Search3`
//!   lazy-2 driver with C's overlapping-match resolution.
//! * `LZ4HC_compress_optimal` (levels 10-12): price-driven parse over a
//!   4096-position window, `sufficient_len` early-out, full update at 12.
//!
//! Indices are `position + HC_BASE` exactly like C's `dictLimit = 64 KiB`
//! initialisation, so `index - delta` never underflows and an empty hash slot
//! (0) is always below the lowest valid match index. The context persists
//! across the blocks of a frame (C's `LZ4_compress_HC_continue`): later
//! blocks may match into the previous 64 KiB, which is what the reference
//! encoder's default *linked* block mode does.

use super::block::{compress_bound, count_match, emit_literal_only, emit_sequence, read_u32, read_u64};

const MINMATCH: usize = 4;
const MFLIMIT: usize = 12;
const LASTLITERALS: usize = 5;
const LZ4_MIN_LENGTH: usize = MFLIMIT + 1;
const DISTANCE_MAX: u32 = 65535;
const ML_MASK: usize = 15;
const RUN_MASK: usize = 15;
const OPTIMAL_ML: usize = ML_MASK - 1 + MINMATCH;
const LZ4_OPT_NUM: usize = 1 << 12;
const TRAILING_LITERALS: usize = 3;
const HASH_LOG: u32 = 15;
const HASH_SIZE: usize = 1 << HASH_LOG;
const CHAIN_SIZE: usize = 1 << 16;
/// `LZ4HC_init_internal` starts indices at 64 KiB.
const HC_BASE: u32 = 1 << 16;

#[derive(Clone, Copy)]
enum Strat {
    Mid,
    HashChain,
    Optimal,
}

/// `k_clTable`: (strategy, nbSearches, targetLength) per level.
fn level_params(level: u32) -> (Strat, i32, usize) {
    match level.min(12) {
        0..=2 => (Strat::Mid, 2, 16),
        3 => (Strat::HashChain, 4, 16),
        4 => (Strat::HashChain, 8, 16),
        5 => (Strat::HashChain, 16, 16),
        6 => (Strat::HashChain, 32, 16),
        7 => (Strat::HashChain, 64, 16),
        8 => (Strat::HashChain, 128, 16),
        9 => (Strat::HashChain, 256, 16),
        10 => (Strat::Optimal, 96, 64),
        11 => (Strat::Optimal, 512, 128),
        _ => (Strat::Optimal, 16384, LZ4_OPT_NUM),
    }
}

#[derive(Clone, Copy, Default)]
struct Match {
    len: usize,
    off: u32,
    /// Bytes the match was extended backwards (C's `back`, negated).
    back: usize,
}

const NOMATCH: Match = Match { len: 0, off: 0, back: 0 };

#[inline(always)]
fn hash_ptr(v: u32) -> usize {
    (v.wrapping_mul(2654435761) >> (MINMATCH as u32 * 8 - HASH_LOG)) as usize
}

/// HC match-finder state, reusable across the blocks of one frame.
pub struct HcCtx {
    hash: Vec<u32>,
    chain: Vec<u16>,
    next_to_update: u32,
    /// Scratch for the optimal parser (allocated on first use).
    opt: Vec<Opt>,
}

impl Default for HcCtx {
    fn default() -> Self {
        Self::new()
    }
}

#[derive(Clone, Copy, Default)]
struct Opt {
    price: i32,
    off: u32,
    mlen: u32,
    litlen: u32,
}

/// `LZ4HC_countBack`: how many bytes before `ip` / `mp` are equal, bounded
/// by `i_min` / `m_min`.
#[inline(always)]
fn count_back(input: &[u8], ip: usize, mp: usize, i_min: usize, m_min: usize) -> usize {
    let min = (ip - i_min).min(mp - m_min);
    let mut back = 0usize;
    while min - back > 3 {
        let v = unsafe { read_u32(input, ip - back - 4) ^ read_u32(input, mp - back - 4) };
        if v != 0 {
            return back + (v.leading_zeros() >> 3) as usize;
        }
        back += 4;
    }
    while back < min && unsafe { *input.get_unchecked(ip - back - 1) == *input.get_unchecked(mp - back - 1) } {
        back += 1;
    }
    back
}

/// `LZ4HC_countPattern`: length of the repeating 1/2/4-byte pattern starting
/// at `ip`, bounded by `i_end`.
fn count_pattern(input: &[u8], mut ip: usize, i_end: usize, pattern32: u32) -> usize {
    let start = ip;
    let pattern = pattern32 as u64 | ((pattern32 as u64) << 32);
    while ip + 8 <= i_end {
        let diff = unsafe { read_u64(input, ip) } ^ pattern;
        if diff != 0 {
            return ip - start + (diff.trailing_zeros() >> 3) as usize;
        }
        ip += 8;
    }
    let mut pb = pattern;
    while ip < i_end && input[ip] == pb as u8 {
        ip += 1;
        pb >>= 8;
    }
    ip - start
}

/// `LZ4HC_reverseCountPattern`: pattern length going backwards from `ip`.
fn reverse_count_pattern(input: &[u8], mut ip: usize, i_low: usize, pattern: u32) -> usize {
    let start = ip;
    while ip >= i_low + 4 {
        if unsafe { read_u32(input, ip - 4) } != pattern {
            break;
        }
        ip -= 4;
    }
    let bytes = pattern.to_le_bytes();
    let mut k = 3usize;
    while ip > i_low {
        if input[ip - 1] != bytes[k] {
            break;
        }
        ip -= 1;
        k = k.wrapping_sub(1) & 3;
    }
    start - ip
}

impl HcCtx {
    pub fn new() -> Self {
        Self {
            hash: vec![0u32; HASH_SIZE],
            chain: vec![0xFFFFu16; CHAIN_SIZE],
            next_to_update: HC_BASE,
            opt: Vec::new(),
        }
    }

    /// `LZ4HC_Insert`: hash every position in `[next_to_update, ip)`.
    #[inline(always)]
    fn insert(&mut self, input: &[u8], ip: usize) {
        let target = ip as u32 + HC_BASE;
        let mut idx = self.next_to_update;
        let hash = self.hash.as_mut_ptr();
        let chain = self.chain.as_mut_ptr();
        while idx < target {
            // SAFETY: idx - HC_BASE < ip <= mflimit, so 4 bytes are readable.
            let h = hash_ptr(unsafe { read_u32(input, (idx - HC_BASE) as usize) });
            unsafe {
                let mut delta = idx - *hash.add(h);
                if delta > DISTANCE_MAX {
                    delta = DISTANCE_MAX;
                }
                *chain.add((idx & 0xFFFF) as usize) = delta as u16;
                *hash.add(h) = idx;
            }
            idx += 1;
        }
        self.next_to_update = target;
    }

    #[inline(always)]
    fn delta(&self, idx: u32) -> u32 {
        unsafe { *self.chain.get_unchecked((idx & 0xFFFF) as usize) as u32 }
    }

    /// `LZ4HC_InsertAndGetWiderMatch` (single prefix, no external dict,
    /// `favorDecSpeed` off). Searches for a match at `ip` longer than
    /// `longest`, allowed to start as early as `i_low` (backward extension)
    /// and to run up to `i_high`.
    #[allow(clippy::too_many_arguments)]
    #[inline(always)]
    fn wider_match(
        &mut self,
        input: &[u8],
        ip: usize,
        i_low: usize,
        i_high: usize,
        mut longest: usize,
        max_attempts: i32,
        pattern_analysis: bool,
        chain_swap: bool,
    ) -> Match {
        let ip_index = ip as u32 + HC_BASE;
        let lowest = if ip_index < HC_BASE + DISTANCE_MAX + 1 { HC_BASE } else { ip_index - DISTANCE_MAX };
        let look_back = ip - i_low;
        let mut nb = max_attempts;
        let mut chain_pos: u32 = 0;
        let pattern = unsafe { read_u32(input, ip) };
        let mut repeat: u8 = 0; // 0 untested, 1 not, 2 confirmed
        let mut src_pattern_len = 0usize;
        let mut offset = 0u32;
        let mut s_back = 0usize;

        self.insert(input, ip);
        let mut match_index = self.hash[hash_ptr(pattern)];

        while match_index >= lowest && nb > 0 {
            let mut match_length = 0usize;
            nb -= 1;
            let mp = (match_index - HC_BASE) as usize;
            // Only candidates that can beat `longest` get the full compare:
            // check the two bytes at the position that would extend it.
            // SAFETY: longest >= look_back + 1 at every call site, so the
            // second read is at >= mp + 1 (inside the input); `i_low +
            // longest + 1 <= i_high + 1` bounds the first.
            let same_tail = unsafe {
                let a = input.as_ptr().add(i_low + longest - 1);
                let b = input.as_ptr().add(mp.wrapping_sub(look_back).wrapping_add(longest - 1));
                core::ptr::read_unaligned(a as *const u16) == core::ptr::read_unaligned(b as *const u16)
            };
            if same_tail && unsafe { read_u32(input, mp) } == pattern {
                let back = if look_back != 0 { count_back(input, ip, mp, i_low, 0) } else { 0 };
                let ml = MINMATCH + count_match(input, mp + MINMATCH, ip + MINMATCH, i_high) + back;
                match_length = ml;
                if ml > longest {
                    longest = ml;
                    offset = ip_index - match_index;
                    s_back = back;
                }
            }

            if chain_swap && match_length == longest {
                // A candidate as long as the current best: hop to the chain
                // whose next link is farthest away (more distinct candidates).
                debug_assert_eq!(look_back, 0);
                if match_index + longest as u32 <= ip_index {
                    const K_TRIGGER: u32 = 4;
                    let mut dist_to_next: u32 = 1;
                    let end = (longest - MINMATCH + 1) as u32;
                    let mut accel: u32 = 1 << K_TRIGGER;
                    let mut pos: u32 = 0;
                    while pos < end {
                        let cd = self.delta(match_index + pos);
                        let step = accel >> K_TRIGGER;
                        accel += 1;
                        if cd > dist_to_next {
                            dist_to_next = cd;
                            chain_pos = pos;
                            accel = 1 << K_TRIGGER;
                        }
                        pos += step;
                    }
                    if dist_to_next > 1 {
                        if dist_to_next > match_index {
                            break;
                        }
                        match_index -= dist_to_next;
                        continue;
                    }
                }
            }

            let dist_next = self.delta(match_index);
            if pattern_analysis && dist_next == 1 && chain_pos == 0 {
                let cand = match_index - 1;
                if repeat == 0 {
                    if (pattern & 0xFFFF) == (pattern >> 16) && (pattern & 0xFF) == (pattern >> 24) {
                        repeat = 2;
                        src_pattern_len = count_pattern(input, ip + 4, i_high, pattern) + 4;
                    } else {
                        repeat = 1;
                    }
                }
                if repeat == 2 && cand >= lowest {
                    let cmp = (cand - HC_BASE) as usize;
                    if unsafe { read_u32(input, cmp) } == pattern {
                        let forward = count_pattern(input, cmp + 4, i_high, pattern) + 4;
                        let back_len = reverse_count_pattern(input, cmp, 0, pattern);
                        let back_len = (cand - (cand - back_len.min(cmp) as u32).max(lowest)) as usize;
                        let segment = back_len + forward;
                        if segment >= src_pattern_len && forward <= src_pattern_len {
                            match_index = cand + forward as u32 - src_pattern_len as u32;
                        } else {
                            match_index = cand - back_len as u32;
                            if look_back == 0 {
                                let max_ml = segment.min(src_pattern_len);
                                if longest < max_ml {
                                    if ip_index - match_index > DISTANCE_MAX {
                                        break;
                                    }
                                    longest = max_ml;
                                    offset = ip_index - match_index;
                                    debug_assert_eq!(s_back, 0);
                                }
                                let d = self.delta(match_index);
                                if d > match_index {
                                    break;
                                }
                                match_index -= d;
                            }
                        }
                        continue;
                    }
                }
            }

            match_index = match_index.wrapping_sub(self.delta(match_index + chain_pos));
        }

        Match { len: longest, off: offset, back: s_back }
    }

    /// `LZ4HC_InsertAndFindBestMatch`.
    #[inline(always)]
    fn find_best(&mut self, input: &[u8], ip: usize, i_high: usize, attempts: i32, pa: bool) -> Match {
        self.wider_match(input, ip, ip, i_high, MINMATCH - 1, attempts, pa, false)
    }

    /// `LZ4HC_FindLongerMatch` (optimal parser).
    #[inline(always)]
    fn find_longer(&mut self, input: &[u8], ip: usize, i_high: usize, min_len: usize, attempts: i32) -> Match {
        let m = self.wider_match(input, ip, ip, i_high, min_len, attempts, true, true);
        debug_assert_eq!(m.back, 0);
        if m.len <= min_len {
            NOMATCH
        } else {
            m
        }
    }
}

struct Emit<'a> {
    input: &'a [u8],
    out: &'a mut Vec<u8>,
    anchor: usize,
}

impl Emit<'_> {
    /// `LZ4HC_encodeSequence`: emits literals `[anchor, ip)` + the match and
    /// advances `ip`/`anchor` past it.
    #[inline(always)]
    fn seq(&mut self, ip: &mut usize, len: usize, off: u32) {
        emit_sequence(self.out, &self.input[self.anchor..*ip], *ip - self.anchor, off as u16, len);
        *ip += len;
        self.anchor = *ip;
    }
}

/// `LZ4HC_compress_hashChain` over `input[block_start..block_end]`; the
/// match finder may reference earlier bytes of `input` (linked blocks).
fn compress_hash_chain(
    ctx: &mut HcCtx,
    input: &[u8],
    block_start: usize,
    block_end: usize,
    out: &mut Vec<u8>,
    attempts: i32,
) {
    let pa = attempts > 128;
    let mflimit = block_end - MFLIMIT;
    let matchlimit = block_end - LASTLITERALS;
    let mut ip = block_start;
    let mut e = Emit { input, out, anchor: block_start };

    'main: while ip <= mflimit {
        let mut m1 = ctx.find_best(input, ip, matchlimit, attempts, pa);
        if m1.len < MINMATCH {
            ip += 1;
            continue;
        }
        let mut start0 = ip;
        let mut m0 = m1;

        // C's `_Search2:` label — every exit from the inner `_Search3` loop
        // either finishes this position (`continue 'main`) or comes back
        // here (`break`).
        loop {
            let mut start2;
            let mut m2;
            if ip + m1.len <= mflimit {
                start2 = ip + m1.len - 2;
                m2 = ctx.wider_match(input, start2, ip, matchlimit, m1.len, attempts, pa, false);
                start2 -= m2.back;
            } else {
                start2 = 0;
                m2 = NOMATCH;
            }
            if m2.len <= m1.len {
                e.seq(&mut ip, m1.len, m1.off);
                continue 'main;
            }
            if start0 < ip && start2 < ip + m0.len {
                // The first match was skipped at least once: restore it.
                ip = start0;
                m1 = m0;
            }
            if start2 - ip < 3 {
                // First match too small: drop it, ML2 becomes ML1.
                ip = start2;
                m1 = m2;
                continue;
            }

            // C's `_Search3:` label.
            loop {
                if start2 - ip < OPTIMAL_ML {
                    let mut new_ml = m1.len.min(OPTIMAL_ML);
                    if ip + new_ml > start2 + m2.len - MINMATCH {
                        new_ml = start2 - ip + m2.len - MINMATCH;
                    }
                    if new_ml > start2 - ip {
                        let correction = new_ml - (start2 - ip);
                        start2 += correction;
                        m2.len -= correction;
                    }
                }

                let mut start3;
                let m3;
                if start2 + m2.len <= mflimit {
                    start3 = start2 + m2.len - 3;
                    m3 = ctx.wider_match(input, start3, start2, matchlimit, m2.len, attempts, pa, false);
                    start3 -= m3.back;
                } else {
                    start3 = 0;
                    m3 = NOMATCH;
                }

                if m3.len <= m2.len {
                    // No better match: encode ML1 and ML2.
                    if start2 < ip + m1.len {
                        m1.len = start2 - ip;
                    }
                    e.seq(&mut ip, m1.len, m1.off);
                    ip = start2;
                    e.seq(&mut ip, m2.len, m2.off);
                    continue 'main;
                }

                if start3 < ip + m1.len + 3 {
                    // Not enough space for match 2: remove it.
                    if start3 >= ip + m1.len {
                        // Seq1 can be written now; Seq3 becomes Seq1.
                        if start2 < ip + m1.len {
                            let correction = ip + m1.len - start2;
                            start2 += correction;
                            m2.len -= correction;
                            if m2.len < MINMATCH {
                                start2 = start3;
                                m2 = m3;
                            }
                        }
                        e.seq(&mut ip, m1.len, m1.off);
                        ip = start3;
                        m1 = m3;
                        start0 = start2;
                        m0 = m2;
                        break; // -> _Search2
                    }
                    start2 = start3;
                    m2 = m3;
                    continue; // -> _Search3
                }

                // Three ascending matches: write ML1, shift the others down.
                if start2 < ip + m1.len {
                    if start2 - ip < OPTIMAL_ML {
                        if m1.len > OPTIMAL_ML {
                            m1.len = OPTIMAL_ML;
                        }
                        if ip + m1.len > start2 + m2.len - MINMATCH {
                            m1.len = start2 - ip + m2.len - MINMATCH;
                        }
                        if m1.len > start2 - ip {
                            let correction = m1.len - (start2 - ip);
                            start2 += correction;
                            m2.len -= correction;
                        }
                    } else {
                        m1.len = start2 - ip;
                    }
                }
                e.seq(&mut ip, m1.len, m1.off);
                ip = start2;
                m1 = m2;
                start2 = start3;
                m2 = m3;
                // -> _Search3
            }
        }
    }

    emit_literal_only(e.out, &input[e.anchor..block_end]);
}


#[inline(always)]
fn literals_price(litlen: usize) -> i32 {
    let mut price = litlen as i32;
    if litlen >= RUN_MASK {
        price += 1 + ((litlen - RUN_MASK) / 255) as i32;
    }
    price
}

#[inline(always)]
fn sequence_price(litlen: usize, mlen: usize) -> i32 {
    let mut price = 1 + 2 + literals_price(litlen);
    if mlen >= ML_MASK + MINMATCH {
        price += 1 + ((mlen - (ML_MASK + MINMATCH)) / 255) as i32;
    }
    price
}

/// `LZ4HC_compress_optimal`.
#[allow(clippy::too_many_arguments, unused_assignments)]
fn compress_optimal(
    ctx: &mut HcCtx,
    input: &[u8],
    block_start: usize,
    block_end: usize,
    out: &mut Vec<u8>,
    nb_searches: i32,
    sufficient_len: usize,
    full_update: bool,
) {
    let sufficient_len = sufficient_len.min(LZ4_OPT_NUM - 1);
    let mflimit = block_end - MFLIMIT;
    let matchlimit = block_end - LASTLITERALS;
    let mut ip = block_start;
    let mut e = Emit { input, out, anchor: block_start };
    if ctx.opt.len() < LZ4_OPT_NUM + TRAILING_LITERALS {
        ctx.opt = vec![Opt::default(); LZ4_OPT_NUM + TRAILING_LITERALS];
    }
    let mut opt = core::mem::take(&mut ctx.opt);

    while ip <= mflimit {
        let llen = ip - e.anchor;
        let first = ctx.find_longer(input, ip, matchlimit, MINMATCH - 1, nb_searches);
        if first.len == 0 {
            ip += 1;
            continue;
        }
        if first.len > sufficient_len {
            e.seq(&mut ip, first.len, first.off);
            continue;
        }

        // Prices for the first positions (literals).
        for r in 0..MINMATCH {
            opt[r] = Opt { mlen: 1, off: 0, litlen: (llen + r) as u32, price: literals_price(llen + r) };
        }
        // Prices using the initial match.
        for mlen in MINMATCH..=first.len {
            opt[mlen] = Opt { mlen: mlen as u32, off: first.off, litlen: llen as u32, price: sequence_price(llen, mlen) };
        }
        let mut last_match_pos = first.len;
        for add in 1..=TRAILING_LITERALS {
            opt[last_match_pos + add] = Opt {
                mlen: 1,
                off: 0,
                litlen: add as u32,
                price: opt[last_match_pos].price + literals_price(add),
            };
        }

        let (mut best_mlen, mut best_off, mut cur) = (0usize, 0u32, 0usize);
        let mut immediate = false;
        cur = 1;
        while cur < last_match_pos {
            let cur_ptr = ip + cur;
            if cur_ptr > mflimit {
                break;
            }
            if full_update {
                if opt[cur + 1].price <= opt[cur].price && opt[cur + MINMATCH].price < opt[cur].price + 3 {
                    cur += 1;
                    continue;
                }
            } else if opt[cur + 1].price <= opt[cur].price {
                cur += 1;
                continue;
            }

            let new_match = if full_update {
                ctx.find_longer(input, cur_ptr, matchlimit, MINMATCH - 1, nb_searches)
            } else {
                ctx.find_longer(input, cur_ptr, matchlimit, last_match_pos - cur, nb_searches)
            };
            if new_match.len == 0 {
                cur += 1;
                continue;
            }

            if new_match.len > sufficient_len || new_match.len + cur >= LZ4_OPT_NUM {
                best_mlen = new_match.len;
                best_off = new_match.off;
                last_match_pos = cur + 1;
                immediate = true;
                break;
            }

            // Literals before the match.
            let base_litlen = opt[cur].litlen as usize;
            for litlen in 1..MINMATCH {
                let price = opt[cur].price - literals_price(base_litlen) + literals_price(base_litlen + litlen);
                let pos = cur + litlen;
                if price < opt[pos].price {
                    opt[pos] = Opt { mlen: 1, off: 0, litlen: (base_litlen + litlen) as u32, price };
                }
            }
            // Prices using the match at `cur`.
            let match_ml = new_match.len;
            for ml in MINMATCH..=match_ml {
                let pos = cur + ml;
                let (ll, price) = if opt[cur].mlen == 1 {
                    let ll = opt[cur].litlen as usize;
                    (ll, (if cur > ll { opt[cur - ll].price } else { 0 }) + sequence_price(ll, ml))
                } else {
                    (0, opt[cur].price + sequence_price(0, ml))
                };
                if pos > last_match_pos + TRAILING_LITERALS || price <= opt[pos].price {
                    if ml == match_ml && last_match_pos < pos {
                        last_match_pos = pos;
                    }
                    opt[pos] = Opt { mlen: ml as u32, off: new_match.off, litlen: ll as u32, price };
                }
            }
            for add in 1..=TRAILING_LITERALS {
                opt[last_match_pos + add] = Opt {
                    mlen: 1,
                    off: 0,
                    litlen: add as u32,
                    price: opt[last_match_pos].price + literals_price(add),
                };
            }
            cur += 1;
        }

        if !immediate {
            best_mlen = opt[last_match_pos].mlen as usize;
            best_off = opt[last_match_pos].off;
            cur = last_match_pos - best_mlen;
        }

        // Reverse traversal: record the chosen path at its start positions.
        {
            let mut candidate_pos = cur;
            let mut sel_len = best_mlen;
            let mut sel_off = best_off;
            loop {
                let next_len = opt[candidate_pos].mlen as usize;
                let next_off = opt[candidate_pos].off;
                opt[candidate_pos].mlen = sel_len as u32;
                opt[candidate_pos].off = sel_off;
                sel_len = next_len;
                sel_off = next_off;
                if next_len > candidate_pos {
                    break;
                }
                candidate_pos -= next_len;
            }
        }
        // Encode the recorded sequences in order.
        let mut r = 0usize;
        while r < last_match_pos {
            let ml = opt[r].mlen as usize;
            let off = opt[r].off;
            if ml == 1 {
                ip += 1;
                r += 1;
                continue;
            }
            r += ml;
            e.seq(&mut ip, ml, off);
        }
    }

    emit_literal_only(e.out, &input[e.anchor..block_end]);
    ctx.opt = opt;
}

/// Compress `input[block_start..block_end]` as one LZ4 block at HC `level`
/// (2..=12, clamped; 2 is `LZ4MID`), appending to `output`. `ctx` carries the match
/// finder across consecutive blocks of the same `input` (linked blocks).
/// Returns the number of bytes appended.
pub fn compress_block_hc_continue(
    ctx: &mut HcCtx,
    input: &[u8],
    block_start: usize,
    block_end: usize,
    output: &mut Vec<u8>,
    level: u32,
) -> usize {
    let start_out = output.len();
    let len = block_end - block_start;
    output.reserve(compress_bound(len));
    if len < LZ4_MIN_LENGTH {
        // Too small for any sequence: all literals. Hashing stays lazy
        // (C leaves `nextToUpdate` alone here too).
        emit_literal_only(output, &input[block_start..block_end]);
        return output.len() - start_out;
    }
    let (strat, nb, target) = level_params(level);
    match strat {
        Strat::Mid => compress_mid(ctx, input, block_start, block_end, output),
        Strat::HashChain => compress_hash_chain(ctx, input, block_start, block_end, output, nb),
        Strat::Optimal => compress_optimal(ctx, input, block_start, block_end, output, nb, target, level >= 12),
    }
    // `LZ4HC_Insert` is lazy; skipping the unhashed tail of this block is
    // what C does as well (`nextToUpdate` only advances on demand).
    output.len() - start_out
}

// =========================================================================
// Level 2: `LZ4MID_compress` (lz4hc.c, lz4 1.10)
// =========================================================================

const MID_HASHLOG: u32 = HASH_LOG - 1;
const MID_TABLE_SIZE: usize = 1 << MID_HASHLOG;
const MID_HASHSIZE: usize = 8;

#[inline(always)]
fn mid_hash4(v: u32) -> usize {
    (v.wrapping_mul(2654435761) >> (32 - MID_HASHLOG)) as usize
}

/// `LZ4MID_hash7`: hashes the low 56 bits of the little-endian 64-bit load.
#[inline(always)]
fn mid_hash8(v: u64) -> usize {
    ((v << 8).wrapping_mul(58295818150454627u64) >> (64 - MID_HASHLOG)) as usize
}

/// `LZ4MID_compress` over `input[block_start..block_end]` (prefix mode: the
/// match finder may reference earlier bytes of `input`, i.e. linked
/// blocks). The two 2^14 tables share `ctx.hash` exactly like C's
/// `hash4Table` / `hash8Table = hash4Table + LZ4MID_HASHTABLESIZE`, and
/// positions are absolute indices offset by `HC_BASE`, so an empty slot (0)
/// is never within `DISTANCE_MAX`.
///
/// Two C quirks are reproduced on purpose because they change the output:
/// `ipIndex` is computed at the loop top and NOT refreshed after the
/// `ip+1` longer-match step or the catch-back, yet the "beginning of match"
/// table fills hash the adjusted `ip` with that stale index.
fn compress_mid(ctx: &mut HcCtx, input: &[u8], block_start: usize, block_end: usize, out: &mut Vec<u8>) {
    let (h4, h8) = ctx.hash.split_at_mut(MID_TABLE_SIZE);
    let mflimit = block_end - MFLIMIT;
    let matchlimit = block_end - LASTLITERALS;
    let ilimit_idx = (block_end - MID_HASHSIZE) as u32 + HC_BASE;
    let mut e = Emit { input, out, anchor: block_start };
    let mut ip = block_start;

    // SAFETY of the unchecked reads below: `ip <= mflimit = block_end - 12`
    // in the main loop, so 8-byte loads at ip, ip+1, ip+2 stay inside the
    // block; the end-of-match fills are guarded by `pos_m2 < ilimit_idx`.
    while ip <= mflimit {
        let ip_index = ip as u32 + HC_BASE;
        let mut match_len;
        let match_dist: u32;

        // Long match candidate.
        let hh8 = mid_hash8(unsafe { read_u64(input, ip) });
        let pos8 = h8[hh8];
        h8[hh8] = ip_index;
        if ip_index.wrapping_sub(pos8) <= DISTANCE_MAX && pos8 >= HC_BASE {
            let mp = (pos8 - HC_BASE) as usize;
            match_len = count_match(input, mp, ip, matchlimit);
            if match_len >= MINMATCH {
                match_dist = ip_index - pos8;
                encode_mid(&mut e, &mut ip, ip_index, match_len, match_dist, h4, h8, ilimit_idx);
                continue;
            }
        }
        // Short match candidate.
        let hh4 = mid_hash4(unsafe { read_u32(input, ip) });
        let pos4 = h4[hh4];
        h4[hh4] = ip_index;
        if ip_index.wrapping_sub(pos4) <= DISTANCE_MAX && pos4 >= HC_BASE {
            let mp = (pos4 - HC_BASE) as usize;
            match_len = count_match(input, mp, ip, matchlimit);
            if match_len >= MINMATCH {
                // Short match found; check ip+1 for a longer one.
                let hh8b = mid_hash8(unsafe { read_u64(input, ip + 1) });
                let pos8b = h8[hh8b];
                let m2_dist = (ip_index + 1).wrapping_sub(pos8b);
                let mut dist = ip_index - pos4;
                if m2_dist <= DISTANCE_MAX && pos8b >= HC_BASE && ip < mflimit {
                    let m2 = (pos8b - HC_BASE) as usize;
                    let ml2 = count_match(input, m2, ip + 1, matchlimit);
                    if ml2 > match_len {
                        h8[hh8b] = ip_index + 1;
                        ip += 1;
                        match_len = ml2;
                        dist = m2_dist;
                    }
                }
                encode_mid(&mut e, &mut ip, ip_index, match_len, dist, h4, h8, ilimit_idx);
                continue;
            }
        }
        // No match: skip faster over incompressible data.
        ip += 1 + ((ip - e.anchor) >> 9);
    }

    emit_literal_only(e.out, &input[e.anchor..block_end]);
}

/// `_lz4mid_encode_sequence`: catch-back, table fills around the match,
/// emission. `ip_index` is the (stale, see above) loop-top index.
#[inline(always)]
#[allow(clippy::too_many_arguments)]
fn encode_mid(
    e: &mut Emit<'_>,
    ip: &mut usize,
    ip_index: u32,
    mut match_len: usize,
    match_dist: u32,
    h4: &mut [u32],
    h8: &mut [u32],
    ilimit_idx: u32,
) {
    let input = e.input;
    let d = match_dist as usize;
    // Catch back.
    while *ip > e.anchor && *ip > d && input[*ip - 1] == input[*ip - d - 1] {
        *ip -= 1;
        match_len += 1;
    }
    // Fill table with the beginning of the match.
    unsafe {
        h8[mid_hash8(read_u64(input, *ip + 1))] = ip_index + 1;
        h8[mid_hash8(read_u64(input, *ip + 2))] = ip_index + 2;
        h4[mid_hash4(read_u32(input, *ip + 1))] = ip_index + 1;
    }
    e.seq(ip, match_len, match_dist);
    // Fill table with the end of the match.
    let end_idx = *ip as u32 + HC_BASE;
    let pos_m2 = end_idx - 2;
    if pos_m2 < ilimit_idx {
        let p = *ip;
        unsafe {
            if p > 5 {
                h8[mid_hash8(read_u64(input, p - 5))] = end_idx - 5;
            }
            h8[mid_hash8(read_u64(input, p - 3))] = end_idx - 3;
            h8[mid_hash8(read_u64(input, p - 2))] = end_idx - 2;
            h4[mid_hash4(read_u32(input, p - 2))] = end_idx - 2;
            h4[mid_hash4(read_u32(input, p - 1))] = end_idx - 1;
        }
    }
}
