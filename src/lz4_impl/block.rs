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
    // SAFETY: per the contract every 32-byte step reads/writes inside the
    // `length + 32` windows, and with `dst - src >= 16` the two ordered
    // 16-byte moves only read bytes written by earlier steps.
    unsafe {
        let end = dst.add(length);
        loop {
            let a = core::ptr::read_unaligned(src as *const [u8; 16]);
            core::ptr::write_unaligned(dst as *mut [u8; 16], a);
            let b = core::ptr::read_unaligned(src.add(16) as *const [u8; 16]);
            core::ptr::write_unaligned(dst.add(16) as *mut [u8; 16], b);
            src = src.add(32);
            dst = dst.add(32);
            if dst >= end {
                return;
            }
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
    // Like `LZ4_decompress_safe`: an empty block is invalid (an empty
    // payload is the 1-byte block `00`).
    if input.is_empty() {
        return Err(io::Error::new(io::ErrorKind::InvalidData, "lz4: empty block"));
    }
    // Cheap floor: output is almost always ≥ input, and the `OUT_SLACK`
    // invariant check grows the Vec geometrically beyond that (glibc
    // realloc is an mremap for large buffers, so growth is nearly free).
    // Reserving a huge worst-case bound here used to cost an mmap/munmap
    // pair per call, which dominated small-block decode time.
    output.reserve(input.len() + OUT_SLACK);

    let start = output.len();
    let in_len = input.len();
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
            // SAFETY: every byte in start..op has been written.
            unsafe { output.set_len(op) };
            return Err(io::Error::new($kind, $msg));
        }};
    }

    while ip < in_len {
        ensure_out!(0);
        let token = input[ip];
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
                let b = input[ip];
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
                let src = &input[ip..ip + length + 32];
                // SAFETY: `src` covers the literals plus the 31-byte
                // over-read; `ensure_out!(length)` left `length + OUT_SLACK`
                // writable bytes at `base + op`.
                unsafe {
                    if length >= 1024 {
                        core::ptr::copy_nonoverlapping(src.as_ptr(), base.add(op), length);
                    } else {
                        wildcopy32(src.as_ptr(), base.add(op), length);
                    }
                }
                ip += length;
                op += length;
                checked_tail = true;
            } else {
                if ip + length > in_len {
                    fail!(io::ErrorKind::UnexpectedEof, "lz4: literal run exceeds input");
                }
                let src = &input[ip..ip + length];
                // SAFETY: `ensure_out!(length)` left `length` writable bytes
                // at `base + op`; `src` is a disjoint input slice.
                unsafe { core::ptr::copy_nonoverlapping(src.as_ptr(), base.add(op), length) };
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
            // SAFETY: `ip + 17 <= in_len` (checked just above) so 16 bytes
            // are readable at `ip`; the loop-top `ensure_out!(0)` guarantees
            // `OUT_SLACK` (64) writable bytes at `base + op`. (A checked
            // `input[ip..ip + 16]` here cost ~5% of decode instructions.)
            unsafe {
                let v = core::ptr::read_unaligned(input.as_ptr().add(ip) as *const [u8; 16]);
                core::ptr::write_unaligned(base.add(op) as *mut [u8; 16], v);
            }
            ip += length;
            op += length;
            checked_tail = true;
        } else {
            if ip + length > in_len {
                fail!(io::ErrorKind::UnexpectedEof, "lz4: literal run exceeds input");
            }
            let src = &input[ip..ip + length];
            // SAFETY: `length <= 14 < OUT_SLACK` writable bytes at
            // `base + op` from the loop-top `ensure_out!(0)`.
            unsafe { core::ptr::copy_nonoverlapping(src.as_ptr(), base.add(op), length) };
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
        // SAFETY: `ip + 2 <= in_len`, either by the check just above or
        // because `checked_tail` came from a path that verified
        // `ip + length + 32 <= in_len` / `ip + 17 <= in_len` before
        // advancing `ip` by `length` (<= 14 on the latter).
        let offset = unsafe { u16::from_le_bytes([byte_at(input, ip), byte_at(input, ip + 1)]) } as usize;
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
                let b = input[ip];
                ip += 1;
                ml += b as usize;
                if b != 255 {
                    break;
                }
            }
            ml += MIN_MATCH;
            ensure_out!(ml);
            // SAFETY: `1 <= offset <= op` so the source is inside the
            // written region; `ensure_out!(ml)` left `ml + OUT_SLACK`
            // writable bytes for the copy and its <= 31-byte overshoot.
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
            // SAFETY: see above; `1 <= offset <= op` keeps the source
            // inside the written region.
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

// =========================================================================
// Fast parser — port of `LZ4_compress_generic_validated` (lz4.c 1.10)
// =========================================================================
//
// Two table flavours, exactly as C picks them:
//   * `byU32` (4096 x u32, `LZ4_hash5` over 8 bytes with the prime
//     889523592379): the streaming path (`LZ4_compress_fast_continue`) that
//     the C frame encoder uses for every block, and the block API for inputs
//     >= 65547 bytes.
//   * `byU16` (8192 x u16, `LZ4_hash4`): the block API (`LZ4_compress_default`)
//     for inputs < 65547 bytes; it also skips the distance check.
// Table entries are absolute input positions (C's `base + index`), an empty
// slot is 0 — a real candidate, as in C. `acceleration` is 1.

/// `LZ4_HASHLOG` for `LZ4_MEMORY_USAGE = 14`.
const HASHLOG: u32 = 12;
const TABLE_U32: usize = 1 << HASHLOG;
const TABLE_U16: usize = 1 << (HASHLOG + 1);
/// `LZ4_skipTrigger`: after 64 misses the search step grows by one.
const SKIP_TRIGGER: u32 = 6;
/// `LZ4_64Klimit`: inputs below this use the u16 table in the block API.
const LZ4_64K_LIMIT: usize = 65_536 + MFLIMIT - 1;
/// `LZ4_minLength`: smaller inputs are emitted as one literal run.
const MIN_LENGTH: usize = MFLIMIT + 1;

/// Hash table flavour for the fast parser. All three accessors are unchecked
/// (a bounds check per access measured 4-7% on this loop).
///
/// # Safety
/// `hash`: `pos + 8 <= input.len()` (u32 table) / `pos + 4 <= input.len()`
/// (u16 table). `get`/`put`: `h` is a value returned by `hash` (so it is
/// below the table size).
trait FastTable {
    const U16: bool;
    unsafe fn hash(input: &[u8], pos: usize) -> usize;
    unsafe fn get(&self, h: usize) -> usize;
    unsafe fn put(&mut self, h: usize, pos: usize);
}

struct TableU32(Vec<u32>);
struct TableU16(Vec<u16>);

impl FastTable for TableU32 {
    const U16: bool = false;
    #[inline(always)]
    unsafe fn hash(input: &[u8], pos: usize) -> usize {
        // SAFETY: per the trait contract.
        let seq = unsafe { read_u64(input, pos) };
        ((seq << 24).wrapping_mul(889_523_592_379) >> (64 - HASHLOG)) as usize
    }
    #[inline(always)]
    unsafe fn get(&self, h: usize) -> usize {
        // SAFETY: `hash` yields `HASHLOG`-bit values; the table has 2^HASHLOG entries.
        unsafe { *self.0.get_unchecked(h) as usize }
    }
    #[inline(always)]
    unsafe fn put(&mut self, h: usize, pos: usize) {
        // SAFETY: as in `get`.
        unsafe { *self.0.get_unchecked_mut(h) = pos as u32 }
    }
}

impl FastTable for TableU16 {
    const U16: bool = true;
    #[inline(always)]
    unsafe fn hash(input: &[u8], pos: usize) -> usize {
        // SAFETY: per the trait contract.
        let seq = unsafe { read_u32(input, pos) };
        (seq.wrapping_mul(2_654_435_761) >> (32 - (HASHLOG + 1))) as usize
    }
    #[inline(always)]
    unsafe fn get(&self, h: usize) -> usize {
        // SAFETY: `hash` yields `HASHLOG + 1`-bit values; the table has 2^(HASHLOG+1) entries.
        unsafe { *self.0.get_unchecked(h) as usize }
    }
    #[inline(always)]
    unsafe fn put(&mut self, h: usize, pos: usize) {
        // SAFETY: as in `get`.
        unsafe { *self.0.get_unchecked_mut(h) = pos as u16 }
    }
}

/// Persistent fast-parser state for linked frame blocks
/// (`LZ4_stream_t` in prefix mode): one u32 table indexed by absolute input
/// position, kept across blocks so a block can match into the previous
/// 64 KiB. `base` is subtracted from stored positions; it only moves when
/// the input passes 2 GiB (C's `LZ4_renormDictT`).
pub struct FastCtx {
    table: TableU32,
    base: usize,
}

impl Default for FastCtx {
    fn default() -> Self {
        Self::new()
    }
}

impl FastCtx {
    pub fn new() -> Self {
        Self { table: TableU32(vec![0; TABLE_U32]), base: 0 }
    }
}

/// Compress `input[start..end]` as one linked block, continuing `ctx`
/// (`LZ4_compress_fast_continue`, prefix mode, acceleration 1). Output is
/// byte-identical to the C frame encoder's blocks. Returns bytes written.
pub fn compress_block_fast_continue(
    ctx: &mut FastCtx,
    input: &[u8],
    start: usize,
    end: usize,
    output: &mut Vec<u8>,
) -> usize {
    // `LZ4_renormDictT`: keep stored positions below 2^31.
    if end - ctx.base > 0x8000_0000 {
        let delta = (start - ctx.base) - (64 << 10);
        for e in ctx.table.0.iter_mut() {
            *e = (*e as usize).saturating_sub(delta) as u32;
        }
        ctx.base += delta;
    }
    let base = ctx.base;
    compress_generic(&mut ctx.table, &input[base..], start - base, end - base, 0, output)
}

/// Compress one independent block (`LZ4_compress_default`): fresh table,
/// u16 flavour below `LZ4_64K_LIMIT` bytes. Returns bytes written.
pub fn compress_block(input: &[u8], output: &mut Vec<u8>) -> usize {
    if input.len() < LZ4_64K_LIMIT {
        let mut t = TableU16(vec![0; TABLE_U16]);
        compress_generic(&mut t, input, 0, input.len(), 0, output)
    } else {
        let mut t = TableU32(vec![0; TABLE_U32]);
        compress_generic(&mut t, input, 0, input.len(), 0, output)
    }
}

/// `LZ4_compress_generic_validated` with `notLimited` output. `low_limit`
/// is the lowest position the backward match extension may reach
/// (`lowLimit`: the dictionary start, i.e. 0 for a linked stream or the
/// block start for an independent block).
#[inline(always)]
fn compress_generic<T: FastTable>(
    t: &mut T,
    input: &[u8],
    start: usize,
    end: usize,
    low_limit: usize,
    output: &mut Vec<u8>,
) -> usize {
    let out_start = output.len();
    output.reserve(compress_bound(end - start));
    let mut anchor = start;

    if end - start >= MIN_LENGTH {
        let mflimit_plus_one = end - MFLIMIT + 1;
        let matchlimit = end - LAST_LITERALS;

        // SAFETY: every unchecked read below is at a position kept below
        // `mflimit_plus_one` (8-byte hashes end 3 bytes before `end`) or at
        // a table entry < the current position; the backward extension
        // stops at `low_limit`/`anchor`; table indices come from `hash`.
        unsafe {
            let mut ip = start;
            t.put(T::hash(input, ip), ip);
            ip += 1;
            let mut forward_h = T::hash(input, ip);

            'main: loop {
                let mut match_pos;
                // ---- Find a match ----
                {
                    let mut forward_ip = ip;
                    let mut step = 1usize;
                    let mut search_match_nb: u32 = 1 << SKIP_TRIGGER;
                    loop {
                        let h = forward_h;
                        let current = forward_ip;
                        let match_index = t.get(h);
                        ip = forward_ip;
                        forward_ip += step;
                        step = (search_match_nb >> SKIP_TRIGGER) as usize;
                        search_match_nb += 1;
                        if forward_ip > mflimit_plus_one {
                            break 'main;
                        }
                        forward_h = T::hash(input, forward_ip);
                        t.put(h, current);
                        if !T::U16 && match_index + MAX_OFFSET < current {
                            continue; // too far
                        }
                        if read_u32(input, match_index) == read_u32(input, ip) {
                            match_pos = match_index;
                            break;
                        }
                    }
                }

                // ---- Catch up ----
                if match_pos > low_limit
                    && byte_at(input, ip - 1) == byte_at(input, match_pos - 1)
                {
                    loop {
                        ip -= 1;
                        match_pos -= 1;
                        if !(ip > anchor
                            && match_pos > low_limit
                            && byte_at(input, ip - 1) == byte_at(input, match_pos - 1))
                        {
                            break;
                        }
                    }
                }

                // ---- Encode literals + match; `_next_match` re-entry ----
                loop {
                    let offset = ip - match_pos;
                    let mcode = count_match(input, match_pos + MIN_MATCH, ip + MIN_MATCH, matchlimit);
                    emit_sequence(output, &input[anchor..ip], ip - anchor, offset as u16, mcode + MIN_MATCH);
                    ip += mcode + MIN_MATCH;
                    anchor = ip;

                    if ip >= mflimit_plus_one {
                        break 'main;
                    }

                    // Fill table with the position before the match end.
                    t.put(T::hash(input, ip - 2), ip - 2);

                    // Test next position.
                    let h = T::hash(input, ip);
                    let match_index = t.get(h);
                    t.put(h, ip);
                    if (T::U16 || match_index + MAX_OFFSET >= ip)
                        && read_u32(input, match_index) == read_u32(input, ip)
                    {
                        match_pos = match_index;
                        continue; // zero-literal sequence
                    }

                    ip += 1;
                    forward_h = T::hash(input, ip);
                    break;
                }
            }
        }
    }

    // ---- Last literals ----
    emit_literal_only(output, &input[anchor..end]);
    output.len() - out_start
}


/// Read a little-endian u32 from `buf` starting at byte position `pos`
/// using a single unaligned load. These unchecked reads are the encoders'
/// one remaining kind of unsafe: a bounds check per hash/compare measured
/// 4-7% on lz4 HC compress.
///
/// # Safety
/// `pos + 4 <= buf.len()`.
#[inline(always)]
pub(super) unsafe fn read_u32(buf: &[u8], pos: usize) -> u32 {
    debug_assert!(pos + 4 <= buf.len());
    // SAFETY: per the contract.
    unsafe { std::ptr::read_unaligned(buf.as_ptr().add(pos) as *const u32).to_le() }
}

/// Read a little-endian u64 from `buf` starting at byte position `pos`
/// using a single unaligned load.
///
/// # Safety
/// `pos + 8 <= buf.len()`.
#[inline(always)]
pub(super) unsafe fn read_u64(buf: &[u8], pos: usize) -> u64 {
    debug_assert!(pos + 8 <= buf.len());
    // SAFETY: per the contract.
    unsafe { std::ptr::read_unaligned(buf.as_ptr().add(pos) as *const u64).to_le() }
}

/// Unchecked byte read.
///
/// # Safety
/// `pos < buf.len()`.
#[inline(always)]
pub(super) unsafe fn byte_at(buf: &[u8], pos: usize) -> u8 {
    debug_assert!(pos < buf.len());
    // SAFETY: per the contract.
    unsafe { *buf.get_unchecked(pos) }
}

/// Count how many consecutive bytes starting at `(input[ms..], input[is..])`
/// are equal, stopping at `limit` (exclusive bound on `is`).  Reads 8 bytes
/// at a time and finds the first differing byte via XOR + trailing-zero.
/// `ms < is` and `limit <= input.len()` (both hold for every caller: `ms`
/// is an earlier match position, `limit` is `matchlimit`).
#[inline(always)]
pub(super) fn count_match(input: &[u8], mut ms: usize, mut is: usize, limit: usize) -> usize {
    debug_assert!(ms < is && limit <= input.len());
    let start = is;
    while is + 8 <= limit {
        // SAFETY: `is + 8 <= limit <= input.len()` and `ms < is`.
        let diff = unsafe { read_u64(input, ms) ^ read_u64(input, is) };
        if diff == 0 {
            ms += 8;
            is += 8;
        } else {
            return (is - start) + (diff.trailing_zeros() as usize >> 3);
        }
    }
    // SAFETY: `is < limit <= input.len()` and `ms < is`.
    while is < limit && unsafe { byte_at(input, ms) == byte_at(input, is) } {
        ms += 1;
        is += 1;
    }
    is - start
}

/// Emit a "literal-only" sequence (no match) — used for the trailing data and
/// for inputs too small to compress.
pub(super) fn emit_literal_only(output: &mut Vec<u8>, literals: &[u8]) {
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
pub(super) fn emit_sequence(output: &mut Vec<u8>, literals: &[u8], lit_len: usize, offset: u16, match_len: usize) {
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

/// HC compress one block (levels 3..=12, clamped) with a fresh context —
/// the single-block API. Frames reuse one [`super::hc::HcCtx`] across
/// blocks so later blocks can match into the previous 64 KiB.
pub fn compress_block_hc(input: &[u8], output: &mut Vec<u8>, level: u32) -> usize {
    let mut ctx = super::hc::HcCtx::new();
    super::hc::compress_block_hc_continue(&mut ctx, input, 0, input.len(), output, level)
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
