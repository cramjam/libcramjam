//! LZ4 block format encoder/decoder.
//!
//! The block format is described at
//! <https://github.com/lz4/lz4/blob/dev/doc/lz4_Block_format.md>.
//!
//! A block is a sequence of *sequences*, each of which contains a literal run
//! followed by a back-reference (match).  The last sequence has only literals.
//!
//! Sequence layout:
//! ```text
//!   token (1 byte) |  literal_len_bytes...  |  literals  |
//!   offset (2 bytes LE)                                  |
//!   match_len_bytes...                                   |
//! ```
//!
//! The token packs `literal_length` (high nibble) and `match_length - 4`
//! (low nibble).  When either value is 15, additional length bytes follow
//! (one or more 255 + final byte 0..254 summed).
//!
//! Constraints (all enforced):
//!   * Min match = 4 bytes.
//!   * Min offset = 1, max offset = 65 535.
//!   * The last 5 bytes of any block are always literals.
//!   * The last match must end at least 12 bytes before the end of the block.

use std::io;

const MIN_MATCH: usize = 4;
const MAX_OFFSET: usize = 65_535;
/// Search-bound margin: the LAST 12 bytes of the block can't START a
/// match.  (Spec: `mflimit = blockEnd - 12`.)
const MFLIMIT: usize = 12;
/// Match-extension bound: matches can extend to within 5 bytes of the
/// block end, but the LAST 5 bytes must remain literals.
const LAST_LITERALS: usize = 5;
const HASH_BITS: usize = 14;
const HASH_SIZE: usize = 1 << HASH_BITS;
const HASH_MASK: usize = HASH_SIZE - 1;
const NONE: u32 = u32::MAX;
/// Skip-step shift: after every `1 << SKIP_TRIGGER` (= 64) consecutive
/// search misses the parser bumps `step` by 1, exponentially skipping
/// ahead through incompressible regions.  Mirrors `LZ4_skipTrigger` in
/// `lz4.c`.
const SKIP_TRIGGER: u32 = 6;

// =========================================================================
// Decoder
// =========================================================================

/// Bytes of output headroom kept ahead of the write cursor at every point
/// in the hot loop (mirrors `FASTLOOP_SAFE_DISTANCE` in lz4.c). Covers the
/// unconditional 16-byte literal copy, the 18-byte match shortcut and the
/// 15-byte wildcopy overshoot.
pub const OUT_SLACK: usize = 64;

/// `LZ4_wildCopy32`: copy in 32-byte steps as two 16-byte moves (so a
/// back-reference with `offset >= 16` stays correct), overshooting by up to
/// 31 bytes. Used for the long-literal / long-match paths, where the
/// per-iteration overhead of 16-byte steps shows up on incompressible or
/// highly repetitive data.
///
/// # Safety
/// `[src, src + length + 32)` readable, `[dst, dst + length + 32)` writable,
/// and `dst - src >= 16` when the regions overlap.
#[inline(always)]
unsafe fn wildcopy32(mut src: *const u8, mut dst: *mut u8, length: usize) {
    let end = unsafe { dst.add(length) };
    loop {
        unsafe {
            let a = core::ptr::read_unaligned(src as *const [u8; 16]);
            core::ptr::write_unaligned(dst as *mut [u8; 16], a);
            let b = core::ptr::read_unaligned(src.add(16) as *const [u8; 16]);
            core::ptr::write_unaligned(dst.add(16) as *mut [u8; 16], b);
            src = src.add(32);
            dst = dst.add(32);
        }
        if dst >= end {
            return;
        }
    }
}

/// Decompress an LZ4 block into `output`.  Returns the number of OUTPUT bytes
/// written.
///
/// Structured after the fast loop of `LZ4_decompress_generic` in lz4.c: raw
/// pointers, one input-bounds check per sequence in the common case
/// (`lit_len < 15` and ≥ 17 bytes of input left ⇒ blind 16-byte literal copy,
/// offset + match nibble readable), the `ml < 15 && offset >= 8` 18-byte
/// match shortcut, and the shared `copy_match_unchecked` kernel otherwise.
/// Output capacity is reserved up front and re-checked only when the
/// `OUT_SLACK` invariant would break (rare), where the Vec simply grows —
/// no separate "safe" decode path is needed.
#[inline(never)]
pub fn decompress_block(input: &[u8], output: &mut Vec<u8>) -> io::Result<usize> {
    // Cheap floor: output is almost always ≥ input, and the `OUT_SLACK`
    // invariant check grows the Vec geometrically beyond that (glibc
    // realloc is an mremap for large buffers, so growth is nearly free).
    // Reserving a huge worst-case bound here used to cost an mmap/munmap
    // pair per call, which dominated small-block decode time.
    output.reserve(input.len() + OUT_SLACK);

    let start = output.len();
    let in_len = input.len();
    let ibase = input.as_ptr();
    let mut ip = 0usize;

    let mut base = output.as_mut_ptr();
    let mut cap = output.capacity();
    let mut op = start;

    // Grow the output so that `cap - op >= need + OUT_SLACK`.
    macro_rules! ensure_out {
        ($need:expr) => {
            let need: usize = $need;
            if cap - op < need + OUT_SLACK {
                // SAFETY: every byte in start..op has been written.
                unsafe { output.set_len(op) };
                output.reserve((need + OUT_SLACK).max(1 << 20));
                base = output.as_mut_ptr();
                cap = output.capacity();
            }
        };
    }
    macro_rules! fail {
        ($kind:expr, $msg:expr) => {{
            unsafe { output.set_len(op) };
            return Err(io::Error::new($kind, $msg));
        }};
    }

    while ip < in_len {
        ensure_out!(0);
        let token = unsafe { *ibase.add(ip) };
        ip += 1;

        // -- Literal run --
        let mut length = (token >> 4) as usize;
        // True when the literal path guaranteed enough trailing input for
        // the offset + match-length nibble without further checks.
        let checked_tail;
        if length == 15 {
            loop {
                if ip >= in_len {
                    fail!(io::ErrorKind::UnexpectedEof, "lz4: unexpected end while reading literal length");
                }
                let b = unsafe { *ibase.add(ip) };
                ip += 1;
                length += b as usize;
                if b != 255 {
                    break;
                }
            }
            ensure_out!(length);
            if ip + length + 32 <= in_len {
                // Source has ≥ 32 bytes past the literals: wildcopy may
                // over-read 31 and the offset/ML are still in bounds.
                // Very long runs (incompressible data) are cheaper as a
                // real memcpy (rep movsb / AVX loops).
                unsafe {
                    if length >= 1024 {
                        core::ptr::copy_nonoverlapping(ibase.add(ip), base.add(op), length);
                    } else {
                        wildcopy32(ibase.add(ip), base.add(op), length);
                    }
                }
                ip += length;
                op += length;
                checked_tail = true;
            } else {
                if ip + length > in_len {
                    fail!(io::ErrorKind::UnexpectedEof, "lz4: literal run exceeds input");
                }
                unsafe { core::ptr::copy_nonoverlapping(ibase.add(ip), base.add(op), length) };
                ip += length;
                op += length;
                if ip == in_len {
                    break;
                }
                checked_tail = false;
            }
        } else if ip + 17 <= in_len {
            // Literals ≤ 14 bytes: blind 16-byte copy, and the 2-byte offset
            // plus the match nibble are readable.
            unsafe {
                let v = core::ptr::read_unaligned(ibase.add(ip) as *const [u8; 16]);
                core::ptr::write_unaligned(base.add(op) as *mut [u8; 16], v);
            }
            ip += length;
            op += length;
            checked_tail = true;
        } else {
            if ip + length > in_len {
                fail!(io::ErrorKind::UnexpectedEof, "lz4: literal run exceeds input");
            }
            unsafe { core::ptr::copy_nonoverlapping(ibase.add(ip), base.add(op), length) };
            ip += length;
            op += length;
            if ip == in_len {
                // Last sequence: literals only.
                break;
            }
            checked_tail = false;
        }

        // -- Match --
        if !checked_tail && ip + 2 > in_len {
            fail!(io::ErrorKind::UnexpectedEof, "lz4: missing match offset");
        }
        let offset = unsafe { u16::from_le_bytes([*ibase.add(ip), *ibase.add(ip + 1)]) } as usize;
        ip += 2;
        // Rejects offset == 0 (wraps to usize::MAX) and offset > bytes produced.
        if offset.wrapping_sub(1) >= op {
            fail!(io::ErrorKind::InvalidData, "lz4: bad match offset");
        }
        let mut ml = (token & 0x0F) as usize;
        if ml == 15 {
            loop {
                if ip >= in_len {
                    fail!(io::ErrorKind::UnexpectedEof, "lz4: unexpected end while reading match length");
                }
                let b = unsafe { *ibase.add(ip) };
                ip += 1;
                ml += b as usize;
                if b != 255 {
                    break;
                }
            }
            ml += MIN_MATCH;
            ensure_out!(ml);
            unsafe {
                let m = base.add(op - offset);
                let d = base.add(op);
                if offset >= 16 {
                    wildcopy32(m, d, ml);
                } else {
                    crate::cpu_features::copy_match_unchecked(m, d, offset, ml);
                }
            }
            op += ml;
        } else {
            ml += MIN_MATCH;
            // `cap - op >= OUT_SLACK - 14 >= 33` here (loop-top invariant
            // minus the ≤14-byte literal), enough for 18 bytes or a
            // ≤18-byte match plus the kernel's 15-byte overshoot.
            unsafe {
                let m = base.add(op - offset);
                let d = base.add(op);
                if offset >= 8 {
                    // Shortcut: 8 + 8 + 2 bytes covers any ml ≤ 18.
                    core::ptr::copy_nonoverlapping(m, d, 8);
                    core::ptr::copy_nonoverlapping(m.add(8), d.add(8), 8);
                    core::ptr::copy_nonoverlapping(m.add(16), d.add(16), 2);
                } else {
                    crate::cpu_features::copy_match_unchecked(m, d, offset, ml);
                }
            }
            op += ml;
        }
    }

    // SAFETY: start..op fully written; op <= cap by the invariant.
    unsafe { output.set_len(op) };
    Ok(op - start)
}

// =========================================================================
// Encoder
// =========================================================================

/// Worst-case compressed size for an `n`-byte input — used to size output buffers.
/// Matches LZ4_compressBound.
pub fn compress_bound(n: usize) -> usize {
    n + n / 255 + 16
}

/// Compress `input` as an LZ4 block.  Falls back to a single literal-only
/// sequence if the input is too small for any match.
///
/// The parser mirrors `LZ4_compress_generic` in `lz4.c`:
///
/// * Hash table is a flat 14-bit `u32` array (`u32::MAX` sentinel = empty).
/// * On a search miss the parser advances by `step` bytes; `step` starts at
///   1 and grows by 1 every 64 consecutive misses (the `LZ4_skipTrigger`
///   pattern).  This is what makes LZ4 fast on incompressible data — we
///   don't probe every byte.
/// * Prefix-check is a single 32-bit load.
/// * Match extension walks 8 bytes at a time via XOR + `trailing_zeros`.
pub fn compress_block(input: &[u8], output: &mut Vec<u8>) -> usize {
    let start_out = output.len();
    let len = input.len();

    // Pre-reserve worst-case bytes so the per-byte writes inside the
    // emit helpers can skip the realloc check.
    output.reserve(compress_bound(len));

    if len < MFLIMIT + MIN_MATCH {
        emit_literal_only(output, input);
        return output.len() - start_out;
    }

    let mut head: Vec<u32> = vec![NONE; HASH_SIZE];
    let mflimit = len - MFLIMIT;
    let matchlimit = len - LAST_LITERALS;

    let mut ip = 0usize;
    let mut anchor = 0usize;

    // SAFETY: throughout this block we maintain `ip < mflimit` and
    // `mp < ip` whenever we read from `input` or `head`.  All reads
    // therefore stay strictly within the bounds checked at function entry.
    unsafe {
        let head_ptr = head.as_mut_ptr();

        // Seed: the very first byte never produces a back-reference.
        let h0 = hash4_lz4_at(input, ip);
        *head_ptr.add(h0) = ip as u32;
        ip += 1;

        'outer: while ip < mflimit {
            // ----- Search phase: find a 4-byte match with skip-step -----
            let mut forward_ip = ip;
            let mut search_match_nb: u32 = 1u32 << SKIP_TRIGGER;
            let mut match_pos: usize;

            loop {
                ip = forward_ip;
                let step = (search_match_nb >> SKIP_TRIGGER) as usize;
                search_match_nb += 1;
                forward_ip = ip + step;
                if forward_ip > mflimit {
                    // No more candidates.  Encode the trailing literals and exit.
                    break 'outer;
                }

                let h = hash4_lz4_at(input, ip);
                let cand = *head_ptr.add(h);
                *head_ptr.add(h) = ip as u32;
                if cand == NONE {
                    continue;
                }
                let mp = cand as usize;
                // ip > mp by construction (we just stored ip and read the
                // PREVIOUS slot value), so dist > 0 always.
                let dist = ip - mp;
                if dist > MAX_OFFSET {
                    continue;
                }
                // Single 32-bit prefix check.
                if read_u32(input, mp) == read_u32(input, ip) {
                    match_pos = mp;
                    break;
                }
            }

            // ----- Walk back: extend match leftward into the literal run -----
            // (Catches matches that start one or more bytes earlier than the
            //  hashed position.)
            while ip > anchor
                && match_pos > 0
                && *input.get_unchecked(match_pos - 1) == *input.get_unchecked(ip - 1)
            {
                ip -= 1;
                match_pos -= 1;
            }
            let dist = ip - match_pos;

            // ----- Forward match extension (8 bytes at a time) -----
            let mlen = MIN_MATCH
                + count_match(input, match_pos + MIN_MATCH, ip + MIN_MATCH, matchlimit);

            // ----- Emit sequence -----
            let lit_len = ip - anchor;
            emit_sequence(output, &input[anchor..ip], lit_len, dist as u16, mlen);

            ip += mlen;
            anchor = ip;
            if ip >= mflimit {
                break;
            }

            // Hash the position right after the match — improves match
            // discovery for the next iteration without extra cost.
            let h = hash4_lz4_at(input, ip);
            *head_ptr.add(h) = ip as u32;
            ip += 1;
        }
    }

    // Trailing literals (always at least LAST_LITERALS bytes).
    if anchor < len {
        emit_literal_only(output, &input[anchor..]);
    }

    output.len() - start_out
}

/// Read a little-endian u32 from `buf` starting at byte position `pos`
/// using a single unaligned load.
///
/// SAFETY: caller must guarantee `pos + 4 <= buf.len()`.
#[inline(always)]
unsafe fn read_u32(buf: &[u8], pos: usize) -> u32 {
    debug_assert!(pos + 4 <= buf.len());
    std::ptr::read_unaligned(buf.as_ptr().add(pos) as *const u32).to_le()
}

/// Read a little-endian u64 from `buf` starting at byte position `pos`
/// using a single unaligned load.
///
/// SAFETY: caller must guarantee `pos + 8 <= buf.len()`.
#[inline(always)]
unsafe fn read_u64(buf: &[u8], pos: usize) -> u64 {
    debug_assert!(pos + 8 <= buf.len());
    std::ptr::read_unaligned(buf.as_ptr().add(pos) as *const u64).to_le()
}

/// Count how many consecutive bytes starting at `(input[ms..], input[is..])`
/// are equal, stopping at `limit` (exclusive bound on `is`).  Reads 8 bytes
/// at a time and finds the first differing byte via XOR + trailing-zero.
#[inline(always)]
fn count_match(input: &[u8], mut ms: usize, mut is: usize, limit: usize) -> usize {
    let start = is;
    while is + 8 <= limit {
        let diff = unsafe { read_u64(input, ms) ^ read_u64(input, is) };
        if diff == 0 {
            ms += 8;
            is += 8;
        } else {
            return (is - start) + (diff.trailing_zeros() as usize >> 3);
        }
    }
    while is < limit && unsafe { *input.get_unchecked(ms) == *input.get_unchecked(is) } {
        ms += 1;
        is += 1;
    }
    is - start
}

/// Emit a "literal-only" sequence (no match) — used for the trailing data and
/// for inputs too small to compress.
fn emit_literal_only(output: &mut Vec<u8>, literals: &[u8]) {
    let lit_len = literals.len();
    let token_lit_part: u8 = if lit_len < 15 { lit_len as u8 } else { 15 };
    output.push(token_lit_part << 4);
    if lit_len >= 15 {
        let mut remaining = lit_len - 15;
        while remaining >= 255 {
            output.push(255);
            remaining -= 255;
        }
        output.push(remaining as u8);
    }
    output.extend_from_slice(literals);
}

/// Emit a full sequence: literal run + match.
fn emit_sequence(output: &mut Vec<u8>, literals: &[u8], lit_len: usize, offset: u16, match_len: usize) {
    let ml_code = match_len - MIN_MATCH;

    // Token byte: high nibble = lit length code, low nibble = match length code.
    let token_lit: u8 = if lit_len < 15 { lit_len as u8 } else { 15 };
    let token_ml: u8 = if ml_code < 15 { ml_code as u8 } else { 15 };
    output.push((token_lit << 4) | token_ml);

    // Extra literal length bytes.
    if lit_len >= 15 {
        let mut remaining = lit_len - 15;
        while remaining >= 255 {
            output.push(255);
            remaining -= 255;
        }
        output.push(remaining as u8);
    }

    // Literals.
    output.extend_from_slice(literals);

    // Offset (2 bytes LE).
    output.extend_from_slice(&offset.to_le_bytes());

    // Extra match length bytes.
    if ml_code >= 15 {
        let mut remaining = ml_code - 15;
        while remaining >= 255 {
            output.push(255);
            remaining -= 255;
        }
        output.push(remaining as u8);
    }
}

/// SAFETY: caller must guarantee `pos + 4 <= input.len()`.
#[inline(always)]
unsafe fn hash4_lz4_at(input: &[u8], pos: usize) -> usize {
    let v = read_u32(input, pos);
    (v.wrapping_mul(2654435761) >> (32 - HASH_BITS)) as usize & HASH_MASK
}

// =========================================================================
// HC encoder (high compression — levels 3..=12 of the lz4 frame format)
// =========================================================================
//
// Architecture (mirrors `lz4hc.c`):
//
//   * `hc_head[hash]` — most recent position whose 4-byte prefix has this hash.
//   * `hc_chain[pos & WINDOW_MASK]` — the position's predecessor with the same
//     hash, forming a singly-linked list of all positions in the current
//     64 KiB window that share a hash bucket.  Older positions naturally
//     fall off when their slot is reused by a newer position.
//
// At each anchor we walk the chain starting from `hc_head[h(ip)]`, capped at
// `chain_depth` candidates (the level knob), and pick the longest extending
// match.  After emitting a match we insert every position spanned by it so
// later searches can find sub-matches.
//
// The lazy step (level ≥ 4) re-runs the search at `ip + 1` and prefers the
// longer of the two matches — this is the single biggest ratio improvement
// over a strict greedy parser.

const HC_HASH_BITS: usize = 15;
const HC_HASH_SIZE: usize = 1 << HC_HASH_BITS;
const HC_HASH_MASK: usize = HC_HASH_SIZE - 1;
const HC_WINDOW_SIZE: usize = 1 << 16; // 64 KiB == MAX_OFFSET + 1
const HC_WINDOW_MASK: usize = HC_WINDOW_SIZE - 1;

/// Per-level chain-walk depth.  Numbers track lz4hc.c's `LZ4HC_clTable`.
#[inline]
fn hc_chain_depth(level: u32) -> usize {
    match level {
        0..=3 => 4,
        4 => 8,
        5 => 16,
        6 => 32,
        7 => 64,
        8 => 128,
        9 => 256,
        10 => 512,
        11 => 1024,
        _ => HC_HASH_SIZE, // level 12+
    }
}

#[inline(always)]
unsafe fn hc_hash4_at(input: &[u8], pos: usize) -> usize {
    let v = read_u32(input, pos);
    (v.wrapping_mul(2654435761) >> (32 - HC_HASH_BITS)) as usize & HC_HASH_MASK
}

struct HcMatchFinder {
    head: Vec<u32>,
    chain: Vec<u32>,
}

impl HcMatchFinder {
    fn new() -> Self {
        Self {
            head: vec![NONE; HC_HASH_SIZE],
            chain: vec![NONE; HC_WINDOW_SIZE],
        }
    }

    /// Record `pos` as the most recent position with its hash, linking
    /// the previous most-recent into its chain slot.
    #[inline(always)]
    fn insert(&mut self, input: &[u8], pos: usize, matchlimit: usize) {
        if pos + MIN_MATCH > matchlimit {
            return;
        }
        let h = unsafe { hc_hash4_at(input, pos) };
        let prev = self.head[h];
        self.head[h] = pos as u32;
        self.chain[pos & HC_WINDOW_MASK] = prev;
    }

    /// Walk the chain at `pos` (up to `max_chain` candidates) and return the
    /// best `(match_pos, match_len)` whose length is ≥ MIN_MATCH, or None.
    fn find_longest(
        &self,
        input: &[u8],
        pos: usize,
        max_chain: usize,
        matchlimit: usize,
    ) -> Option<(usize, usize)> {
        if pos + MIN_MATCH > matchlimit {
            return None;
        }
        let h = unsafe { hc_hash4_at(input, pos) };
        let mut cand = self.head[h];
        let mut best_len = 0usize;
        let mut best_pos = 0usize;
        let mut tried = 0usize;
        let min_pos = pos.saturating_sub(MAX_OFFSET);

        while cand != NONE && tried < max_chain {
            let mp = cand as usize;
            if mp < min_pos || mp >= pos {
                // Out of window or stale chain entry pointing forward.
                break;
            }
            // Cheap 32-bit prefix probe before walking the full match.
            if unsafe { read_u32(input, mp) == read_u32(input, pos) } {
                let len = MIN_MATCH
                    + count_match(input, mp + MIN_MATCH, pos + MIN_MATCH, matchlimit);
                if len > best_len {
                    best_len = len;
                    best_pos = mp;
                    // Long enough to stop early — extending further is rare
                    // payoff per cycle.
                    if len >= 256 {
                        break;
                    }
                }
            }
            cand = self.chain[mp & HC_WINDOW_MASK];
            tried += 1;
        }

        if best_len >= MIN_MATCH {
            Some((best_pos, best_len))
        } else {
            None
        }
    }
}

/// HC compress one block.  `level` selects the chain-walk depth (see
/// `hc_chain_depth`).  Levels < 3 are routed to `compress_block` instead;
/// levels > 12 are clamped to the deepest search.
pub fn compress_block_hc(input: &[u8], output: &mut Vec<u8>, level: u32) -> usize {
    let start_out = output.len();
    let len = input.len();
    output.reserve(compress_bound(len));

    if len < MFLIMIT + MIN_MATCH {
        emit_literal_only(output, input);
        return output.len() - start_out;
    }

    let mflimit = len - MFLIMIT;
    let matchlimit = len - LAST_LITERALS;
    let max_chain = hc_chain_depth(level);
    let lazy = level >= 4;

    let mut mf = HcMatchFinder::new();
    let mut anchor = 0usize;
    let mut ip = 0usize;

    while ip < mflimit {
        // ----- Find longest match at ip -----
        let m0 = mf.find_longest(input, ip, max_chain, matchlimit);
        mf.insert(input, ip, matchlimit);

        let (mut match_ip, mut match_pos, mut match_len) = match m0 {
            Some((mp, ml)) => (ip, mp, ml),
            None => {
                ip += 1;
                continue;
            }
        };

        // ----- Lazy step: see if (ip+1) gives a longer match -----
        if lazy && match_ip + 1 < mflimit {
            let m1 = mf.find_longest(input, match_ip + 1, max_chain, matchlimit);
            // Insert ip+1 either way so its hash is recorded.
            mf.insert(input, match_ip + 1, matchlimit);
            if let Some((mp1, ml1)) = m1 {
                // Per lz4hc.c: the lazy match wins only if it's strictly longer.
                if ml1 > match_len {
                    match_ip += 1;
                    match_pos = mp1;
                    match_len = ml1;
                }
            }
        }

        // ----- Walk back into the literal run -----
        let mut start_ip = match_ip;
        let mut start_mp = match_pos;
        while start_ip > anchor
            && start_mp > 0
            && unsafe { *input.get_unchecked(start_mp - 1) == *input.get_unchecked(start_ip - 1) }
        {
            start_ip -= 1;
            start_mp -= 1;
            match_len += 1;
        }
        let dist = start_ip - start_mp;
        let lit_len = start_ip - anchor;

        emit_sequence(output, &input[anchor..start_ip], lit_len, dist as u16, match_len);

        // ----- Insert every covered position so future searches see them -----
        // The match occupies bytes [start_ip, start_ip + match_len) — note
        // that match_len has been bumped by the walk-back, so the end of
        // the match is start_ip + match_len, NOT match_ip + match_len.
        let new_ip = start_ip + match_len;
        let mut p = start_ip + 1;
        while p < new_ip && p + MIN_MATCH <= matchlimit {
            mf.insert(input, p, matchlimit);
            p += 1;
        }

        ip = new_ip;
        anchor = ip;
    }

    if anchor < len {
        emit_literal_only(output, &input[anchor..]);
    }

    output.len() - start_out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn block_roundtrip_short() {
        let input = b"hello world";
        let mut compressed = Vec::new();
        compress_block(input, &mut compressed);
        let mut decoded = Vec::new();
        decompress_block(&compressed, &mut decoded).unwrap();
        assert_eq!(decoded.as_slice(), input);
    }

    #[test]
    fn block_roundtrip_long_repeating() {
        let input: Vec<u8> = b"abcdefghijklmnop".repeat(1000);
        let mut compressed = Vec::new();
        compress_block(&input, &mut compressed);
        let mut decoded = Vec::new();
        decompress_block(&compressed, &mut decoded).unwrap();
        assert_eq!(decoded, input);
        // Should compress significantly.
        assert!(compressed.len() < input.len() / 4);
    }

    #[test]
    fn block_roundtrip_random() {
        let mut s: u32 = 0xCAFE_BABE;
        let input: Vec<u8> = (0..8000)
            .map(|_| {
                s ^= s << 13;
                s ^= s >> 17;
                s ^= s << 5;
                (s >> 16) as u8
            })
            .collect();
        let mut compressed = Vec::new();
        compress_block(&input, &mut compressed);
        let mut decoded = Vec::new();
        decompress_block(&compressed, &mut decoded).unwrap();
        assert_eq!(decoded, input);
    }
}
