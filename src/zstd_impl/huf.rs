//! Huffman coding for Zstandard (RFC 8878 Section 4.2).
//!
//! Zstd Huffman uses weight-based tree description (not code lengths like DEFLATE)
//! and supports 1-stream or 4-stream parallel decoding for literals.

use std::io;

use super::bits::ReverseBitReader;
use super::fse;

/// Maximum Huffman table log (number of bits for table lookup).
const HUF_MAX_TABLE_LOG: u32 = 12;

/// Huffman decoding table entry. `repr(C)` so the fast decoder can read it
/// as a little-endian `u16` = `symbol | num_bits << 8`.
#[derive(Clone, Copy, Default)]
#[repr(C)]
struct HufEntry {
    symbol: u8,
    num_bits: u8,
}

/// Huffman decoder.
pub struct HufTable {
    table: Vec<HufEntry>,
    max_bits: u32,
}

impl HufTable {
    /// Build from weights (as decoded from the header).
    /// Weight 0 means symbol not present. Weight w means code length = max_bits + 1 - w.
    /// The implicit last weight (for symbol == weights.len()) is inferred from the
    /// constraint that the sum of (1 << (w-1)) over all weights equals 2^max_bits.
    pub fn from_weights(weights: &[u8]) -> io::Result<Self> {
        if weights.is_empty() || weights.len() > 255 {
            return Err(io::Error::new(io::ErrorKind::InvalidData, "zstd: invalid Huffman weight count"));
        }
        for &w in weights {
            if w as u32 > HUF_MAX_TABLE_LOG {
                return Err(io::Error::new(io::ErrorKind::InvalidData, "zstd: Huffman weight exceeds limit"));
            }
        }

        // Sum of (1 << (w - 1)) for all explicitly-given symbols with w > 0.
        let weight_sum: u32 = weights
            .iter()
            .filter(|&&w| w > 0)
            .map(|&w| 1u32 << (w as u32 - 1))
            .sum();
        if weight_sum == 0 {
            return Err(io::Error::new(io::ErrorKind::InvalidData, "zstd: all Huffman weights are zero"));
        }

        // max_bits is the smallest integer such that 2^max_bits > weight_sum.
        // i.e. max_bits = highest_bit_set(weight_sum), where highest_bit_set is 1-indexed.
        let max_bits = highest_bit(weight_sum) + 1;
        if max_bits > HUF_MAX_TABLE_LOG {
            return Err(io::Error::new(io::ErrorKind::InvalidData, "zstd: Huffman table log too large"));
        }

        // Compute the implicit last weight from the leftover.
        let leftover = (1u32 << max_bits) - weight_sum;
        if !leftover.is_power_of_two() {
            return Err(io::Error::new(io::ErrorKind::InvalidData, "zstd: Huffman leftover is not a power of two"));
        }
        let last_weight = highest_bit(leftover) + 1; // 1-indexed log2 of leftover

        // Per-symbol code length (num_bits).  Symbols with weight 0 are unused.
        let total_symbols = weights.len() + 1;
        let mut num_bits_per_symbol = vec![0u8; total_symbols];
        for (sym, &w) in weights.iter().enumerate() {
            if w > 0 {
                num_bits_per_symbol[sym] = max_bits as u8 + 1 - w;
            }
        }
        num_bits_per_symbol[weights.len()] = max_bits as u8 + 1 - last_weight as u8;

        let table_size = 1usize << max_bits;
        let mut table = vec![HufEntry::default(); table_size];

        // Zstd canonical ordering (RFC 8878): longest codes occupy the LOW
        // end of the table, shortest codes occupy the HIGH end.  Within a
        // single code length, symbols are placed in ascending symbol order.
        //
        // Compute the starting table index for each code length using bit-rank
        // counts (matches the C reference and ruzstd).
        let mut bit_rank = vec![0u32; max_bits as usize + 1];
        for &b in &num_bits_per_symbol {
            if b > 0 {
                bit_rank[b as usize] += 1;
            }
        }
        let mut rank_start = vec![0usize; max_bits as usize + 1];
        // rank_start[max_bits] = 0 (longest codes first); going to shorter codes
        // we add the slots used by the previous (longer) ranks.
        for bits in (1..=max_bits as usize).rev() {
            let prev = if bits == max_bits as usize {
                0
            } else {
                rank_start[bits + 1] + bit_rank[bits + 1] as usize * (1 << (max_bits as usize - (bits + 1)))
            };
            rank_start[bits] = prev;
        }

        let mut rank_pos = rank_start.clone();
        for (sym, &b) in num_bits_per_symbol.iter().enumerate() {
            if b == 0 {
                continue;
            }
            let len = 1usize << (max_bits as u8 - b);
            let base = rank_pos[b as usize];
            for j in 0..len {
                table[base + j] = HufEntry { symbol: sym as u8, num_bits: b };
            }
            rank_pos[b as usize] += len;
        }

        Ok(Self { table, max_bits })
    }

    /// Decode Huffman tree description from the compressed stream.
    ///
    /// Returns the table and the number of bytes consumed from the input.
    pub fn decode_from_stream(data: &[u8]) -> io::Result<(Self, usize)> {
        if data.is_empty() {
            return Err(io::Error::new(io::ErrorKind::InvalidData, "zstd: empty Huffman header"));
        }

        let header_byte = data[0];
        if header_byte < 128 {
            // FSE-compressed weights.
            let compressed_size = header_byte as usize;
            if 1 + compressed_size > data.len() {
                return Err(io::Error::new(io::ErrorKind::InvalidData, "zstd: Huffman header truncated"));
            }
            let weights = decode_weights_fse(&data[1..1 + compressed_size])?;
            let table = Self::from_weights(&weights)?;
            Ok((table, 1 + compressed_size))
        } else {
            // Direct representation: weights are packed as 4-bit values.
            let num_symbols = (header_byte as usize) - 127;
            let num_bytes = (num_symbols + 1) / 2;
            if 1 + num_bytes > data.len() {
                return Err(io::Error::new(io::ErrorKind::InvalidData, "zstd: Huffman weights truncated"));
            }
            let mut weights = Vec::with_capacity(num_symbols);
            for i in 0..num_symbols {
                let byte = data[1 + i / 2];
                let w = if i % 2 == 0 { byte >> 4 } else { byte & 0x0F };
                weights.push(w);
            }
            let table = Self::from_weights(&weights)?;
            Ok((table, 1 + num_bytes))
        }
    }

    /// Decode a single symbol from a backward bitstream.
    #[inline(always)]
    pub fn decode_symbol(&self, bits: &mut ReverseBitReader) -> u8 {
        bits.ensure_bits(self.max_bits);
        let idx = bits.peek_bits(self.max_bits);
        let entry = self.table[idx as usize];
        bits.consume(entry.num_bits as u32);
        entry.symbol
    }

    /// Decode `n` symbols into `out`, using batched bit-reader refills so
    /// the inner loop only checks for refill every 5 symbols instead of
    /// every symbol.  With `max_bits ≤ 11` we can safely consume up to
    /// `5 * 11 = 55` bits per refill (the reader has a 64-bit container).
    #[inline(always)]
    pub fn decode_symbols_into(
        &self,
        bits: &mut ReverseBitReader,
        out: &mut [u8],
    ) {
        let max_bits = self.max_bits;
        let table = &self.table[..];
        let n = out.len();
        let mut i = 0usize;

        // Fast path: 5 symbols per refill check.
        // Requires max_bits * 5 <= 56.  When max_bits > 11 this path is skipped.
        if max_bits * 5 <= 56 {
            let batch_bits = max_bits * 5;
            while i + 5 <= n {
                bits.ensure_bits(batch_bits);
                let chunk = &mut out[i..i + 5];
                for slot in chunk {
                    let e = table[bits.peek_bits(max_bits) as usize];
                    bits.consume(e.num_bits as u32);
                    *slot = e.symbol;
                }
                i += 5;
            }
        }

        // Tail: one symbol at a time.
        while i < n {
            out[i] = self.decode_symbol(bits);
            i += 1;
        }
    }

    /// Fast 4-stream decode (the shape of C zstd's
    /// `HUF_decompress4X1_usingDTable_internal_fast`): four pre-shifted
    /// 64-bit containers carrying a sentinel 1-bit that tracks consumption
    /// (`trailing_zeros` = bits consumed since the last load), 5 symbols per
    /// stream per iteration, one unconditional refill per stream per
    /// iteration, and no per-symbol checks — the iteration count is bounded
    /// up front by the smallest remaining output / input of any stream.
    /// Stops with every reader re-synced so the per-symbol tail finishes.
    ///
    /// # Safety
    /// `out_ptr` must be valid for writes of `end[3]` bytes, with
    /// `pos[i] <= end[i] <= pos[i + 1]`; readers must have had their padding
    /// skipped. (Unchecked on purpose: this is the literal-decoding hot loop,
    /// at 1.2 GB/s a per-symbol check is not free.)
    #[inline(never)]
    unsafe fn decode_4stream_fast(
        &self,
        readers: &mut [ReverseBitReader; 4],
        out_ptr: *mut u8,
        pos: &mut [usize; 4],
        end: &[usize; 4],
    ) {
        let shift = 64 - self.max_bits;
        // `HufEntry` is `repr(C) { symbol: u8, num_bits: u8 }`, so each entry
        // read as a little-endian u16 is `symbol | num_bits << 8`.
        let table = self.table.as_ptr() as *const u16;
        let mut acc = [0u64; 4];
        let mut cur = [core::ptr::null::<u8>(); 4];
        let mut lo = [core::ptr::null::<u8>(); 4];
        for i in 0..4 {
            if !readers[i].fast_ok() {
                return;
            }
            let (index, consumed, container, src) = readers[i].raw_parts();
            acc[i] = (container | 1) << consumed;
            // SAFETY: `index + 8 <= source.len()` (`fast_ok`), so `src + index`
            // is inside the stream.
            cur[i] = unsafe { src.add(index) };
            lo[i] = src;
        }
        let [mut p0, mut p1, mut p2, mut p3] = *pos;
        let [mut a0, mut a1, mut a2, mut a3] = acc;
        let [mut c0, mut c1, mut c2, mut c3] = cur;

        loop {
            // Each iteration emits 5 symbols per stream and moves each cursor
            // down by at most 7 bytes (5 * 12 bits < 64) at its refill; the
            // final re-sync below moves it down by up to 7 more, so budget
            // that too — the cursor must never go below the stream start.
            let mut iters = usize::MAX;
            for i in 0..4 {
                let p = [p0, p1, p2, p3][i];
                let c = [c0, c1, c2, c3][i];
                iters = iters.min((end[i] - p) / 5);
                // SAFETY: `c` only ever moves down from `src + index` and
                // the iteration budget below keeps it `>= lo[i]`, so both
                // pointers are in the same stream.
                let ahead = unsafe { c.offset_from(lo[i]) } as usize;
                iters = iters.min(ahead.saturating_sub(7) / 7);
            }
            if iters == 0 {
                break;
            }
            for _ in 0..iters {
                macro_rules! refill {
                    ($a:ident, $c:ident) => {{
                        let ctz = $a.trailing_zeros();
                        // SAFETY: one iteration consumes < 64 bits per stream,
                        // so `ctz >> 3 <= 7`; `iters` was bounded so the
                        // cursor stays >= `lo` with 8 readable bytes above it.
                        $c = unsafe { $c.sub((ctz >> 3) as usize) };
                        // SAFETY: `[c, c + 8)` is inside the stream (above).
                        let raw = unsafe { u64::from_le(core::ptr::read_unaligned($c as *const u64)) };
                        $a = (raw | 1) << (ctz & 7);
                    }};
                }
                refill!(a0, c0);
                refill!(a1, c1);
                refill!(a2, c2);
                refill!(a3, c3);
                macro_rules! sym {
                    ($a:ident, $p:ident, $k:expr) => {{
                        // SAFETY: `acc >> (64 - max_bits)` is < `1 << max_bits`
                        // = `table.len()`.
                        let e = u16::from_le(unsafe { *table.add(($a >> shift) as usize) });
                        $a <<= e >> 8;
                        // SAFETY: `iters` was bounded by `(end - p) / 5`, so
                        // `p + k < end` for this stream (caller contract on
                        // `out_ptr`).
                        unsafe { *out_ptr.add($p + $k) = e as u8 };
                    }};
                }
                sym!(a0, p0, 0); sym!(a1, p1, 0); sym!(a2, p2, 0); sym!(a3, p3, 0);
                sym!(a0, p0, 1); sym!(a1, p1, 1); sym!(a2, p2, 1); sym!(a3, p3, 1);
                sym!(a0, p0, 2); sym!(a1, p1, 2); sym!(a2, p2, 2); sym!(a3, p3, 2);
                sym!(a0, p0, 3); sym!(a1, p1, 3); sym!(a2, p2, 3); sym!(a3, p3, 3);
                sym!(a0, p0, 4); sym!(a1, p1, 4); sym!(a2, p2, 4); sym!(a3, p3, 4);
                p0 += 5;
                p1 += 5;
                p2 += 5;
                p3 += 5;
            }
        }

        // Re-sync the readers: consumed = sentinel position; the container
        // is re-read (the sentinel may have clobbered bit 0).
        let accs = [a0, a1, a2, a3];
        let curs = [c0, c1, c2, c3];
        for i in 0..4 {
            let ctz = accs[i].trailing_zeros();
            // SAFETY: the extra 7 bytes budgeted in `iters` cover this final
            // move, so `c >= lo[i]` and 8 bytes are readable at `c`.
            let c = unsafe { curs[i].sub((ctz >> 3) as usize) };
            debug_assert!(c >= lo[i], "huffman fast loop ran past the stream start");
            // SAFETY: same stream, `c >= lo[i]`.
            let index = unsafe { c.offset_from(lo[i]) } as usize;
            // SAFETY: `[c, c + 8)` is inside the stream.
            let raw = unsafe { u64::from_le(core::ptr::read_unaligned(c as *const u64)) };
            readers[i].set_raw_parts(index, ctz & 7, raw);
        }
        *pos = [p0, p1, p2, p3];
    }

}

/// Decode Huffman weights using a 2-state FSE-compressed backward bitstream
/// (RFC 8878 Section 4.2.1.1).  Two interleaved decoder states share one
/// distribution table; symbols are emitted alternately until the bitstream
/// is exhausted.
fn decode_weights_fse(data: &[u8]) -> io::Result<Vec<u8>> {
    let mut reader = super::bits::ForwardByteReader::new(data);
    let fse_table = fse::FseTable::decode_table(&mut reader, 255, 6)?;
    let remaining = data.get(reader.position()..).ok_or_else(|| {
        io::Error::new(io::ErrorKind::InvalidData, "zstd: Huffman weight FSE table overran input")
    })?;

    let mut bits = ReverseBitReader::new(remaining)?;
    bits.skip_padding_bits()?;

    let mut state1 = bits.get_bits(fse_table.accuracy_log);
    let mut state2 = bits.get_bits(fse_table.accuracy_log);

    let mut weights = Vec::with_capacity(64);
    // Loop: emit, then update.  When `update` would overrun, the OTHER state
    // still holds one final unread symbol.
    loop {
        weights.push(fse_table.symbol(state1));
        state1 = fse_table.next_state(state1, &mut bits);
        if bits.bits_remaining() < 0 {
            weights.push(fse_table.symbol(state2));
            break;
        }

        weights.push(fse_table.symbol(state2));
        state2 = fse_table.next_state(state2, &mut bits);
        if bits.bits_remaining() < 0 {
            weights.push(fse_table.symbol(state1));
            break;
        }

        if weights.len() > 255 {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "zstd: too many Huffman weights",
            ));
        }
    }
    Ok(weights)
}

/// Decode 1-stream Huffman literals into a caller-provided buffer of
/// exactly the regenerated size. Uses the bulk `decode_symbols_into` path
/// which batches 5 symbols per bit-reader refill check.
#[inline(never)]
pub fn decode_literals_1stream_into(
    table: &HufTable,
    data: &[u8],
    output: &mut [u8],
) -> io::Result<()> {
    let mut bits = ReverseBitReader::new(data)?;
    bits.skip_padding_bits()?;
    table.decode_symbols_into(&mut bits, output);
    Ok(())
}

/// Decode 4-stream Huffman literals into a caller-provided buffer of
/// exactly `regen_size` bytes.
///
/// Decodes all 4 streams in lockstep into a single pre-allocated buffer,
/// exposing memory-level parallelism so the CPU can overlap the 4
/// table-load → shift → store dependency chains that serialize within a
/// single stream. A sequential 4-stream walk would only reach ~25% of the
/// interleaved throughput on modern OOO cores.
#[inline(never)]
pub fn decode_literals_4stream_into(
    table: &HufTable,
    data: &[u8],
    output: &mut [u8],
) -> io::Result<()> {
    if data.len() < 6 {
        return Err(io::Error::new(io::ErrorKind::InvalidData, "zstd: 4-stream Huffman header too short"));
    }
    let s1_size = u16::from_le_bytes([data[0], data[1]]) as usize;
    let s2_size = u16::from_le_bytes([data[2], data[3]]) as usize;
    let s3_size = u16::from_le_bytes([data[4], data[5]]) as usize;
    let s_data = &data[6..];

    let s1_end = s1_size;
    let s2_end = s1_end + s2_size;
    let s3_end = s2_end + s3_size;
    if s3_end > s_data.len() {
        return Err(io::Error::new(io::ErrorKind::InvalidData, "zstd: 4-stream jump table overflows data"));
    }

    let regen_size = output.len();
    let seg_size = (regen_size + 3) / 4;
    // Sanity: first 3 segments are exactly `seg_size`; the last is the
    // remainder. Any degenerate `regen_size < 4` hits the guarded
    // `saturating_sub` path below and the interleaved loop degrades to
    // its tail.
    let last_size = regen_size.saturating_sub(seg_size.saturating_mul(3));
    // Short-regen guard: if seg_size == 0 the output is tiny — fall back
    // to a single-stream-per-stream tail-only decode.
    if seg_size == 0 {
        // Nothing to do; output is empty or near-empty.
        return Ok(());
    }

    let mut r1 = ReverseBitReader::new(&s_data[..s1_end])?;
    let mut r2 = ReverseBitReader::new(&s_data[s1_end..s2_end])?;
    let mut r3 = ReverseBitReader::new(&s_data[s2_end..s3_end])?;
    let mut r4 = ReverseBitReader::new(&s_data[s3_end..])?;
    r1.skip_padding_bits()?;
    r2.skip_padding_bits()?;
    r3.skip_padding_bits()?;
    r4.skip_padding_bits()?;

    let mut readers = [r1, r2, r3, r4];
    let mut pos = [0, seg_size, seg_size * 2, seg_size * 3];
    let end = [seg_size, seg_size * 2, seg_size * 3, seg_size * 3 + last_size];
    // SAFETY: `output` has exactly `regen_size = seg_size * 3 + last_size =
    // end[3]` bytes and the segments are contiguous and in order; the
    // readers had their padding skipped above.
    unsafe { table.decode_4stream_fast(&mut readers, output.as_mut_ptr(), &mut pos, &end) };
    // Tail: each stream to completion, one symbol at a time.
    for i in 0..4 {
        while pos[i] < end[i] {
            output[pos[i]] = table.decode_symbol(&mut readers[i]);
            pos[i] += 1;
        }
    }
    Ok(())
}

#[inline(always)]
fn highest_bit(v: u32) -> u32 {
    debug_assert!(v > 0);
    31 - v.leading_zeros()
}

// =========================================================================
// Huffman ENCODER
// =========================================================================

/// Compute optimal length-limited Huffman code lengths using the package-merge
/// algorithm (Larmore & Hirschberg, 1990).
///
/// `sorted_freqs`: symbol frequencies sorted ascending, all > 0.
/// `max_length`: maximum allowed code length (bits).
///
/// Returns bit lengths in the same order as `sorted_freqs`.
/// The resulting code is complete (Kraft sum = 1) and optimal (minimum
/// weighted path length subject to the length constraint).
#[cfg(test)]
pub(crate) fn package_merge_code_lengths(sorted_freqs: &[u32], max_length: usize) -> Vec<u8> {
    let n = sorted_freqs.len();
    debug_assert!(n >= 2);
    debug_assert!(n <= (1 << max_length), "too many symbols for max_length");

    let fresh: Vec<u64> = sorted_freqs.iter().map(|&f| f as u64).collect();

    // Build sorted item lists at each level.
    //
    // Level 0 contains only the n fresh (single-symbol) items.  Each
    // subsequent level merges n fresh items with *packages* formed by
    // pairing consecutive items from the previous level.  Both sources
    // are already sorted, so we merge in O(n).
    let mut levels: Vec<Vec<(u64, bool)>> = Vec::with_capacity(max_length);

    levels.push(fresh.iter().map(|&f| (f, false)).collect());

    for l in 1..max_length {
        let prev = &levels[l - 1];
        let num_pkg = prev.len() / 2;

        // Packages: pair items [0]+[1], [2]+[3], … from the previous level.
        // Because `prev` is sorted, packages are also sorted (a[i]+a[i+1] is
        // non-decreasing for non-decreasing a).
        let mut packages: Vec<(u64, bool)> = Vec::with_capacity(num_pkg);
        for i in 0..num_pkg {
            packages.push((prev[2 * i].0 + prev[2 * i + 1].0, true));
        }

        // Two-pointer merge of fresh items (sorted) and packages (sorted).
        let mut merged = Vec::with_capacity(n + num_pkg);
        let (mut fi, mut pi) = (0usize, 0usize);
        while fi < n && pi < num_pkg {
            if fresh[fi] <= packages[pi].0 {
                merged.push((fresh[fi], false));
                fi += 1;
            } else {
                merged.push(packages[pi]);
                pi += 1;
            }
        }
        while fi < n {
            merged.push((fresh[fi], false));
            fi += 1;
        }
        while pi < num_pkg {
            merged.push(packages[pi]);
            pi += 1;
        }

        levels.push(merged);
    }

    // Determine code lengths by walking levels from coarsest to finest.
    //
    // At the final level we select the 2*(n-1) cheapest items.  For each
    // selected *fresh* item, the corresponding symbol (in frequency order)
    // gets +1 to its code length.  For each selected *package*, we recurse:
    // expand it into 2 items at the next-finer level.
    let mut bit_lengths = vec![0u8; n];
    let mut to_select = 2 * (n - 1);

    for l in (0..max_length).rev() {
        let level = &levels[l];
        let actual = to_select.min(level.len());

        let mut fresh_count = 0usize;
        let mut pkg_count = 0usize;
        for i in 0..actual {
            if level[i].1 {
                pkg_count += 1;
            } else {
                fresh_count += 1;
            }
        }

        // The cheapest `fresh_count` symbols (in sorted order) get +1 bit.
        for i in 0..fresh_count {
            bit_lengths[i] += 1;
        }

        // Each selected package expands to 2 items at the finer level.
        to_select = 2 * pkg_count;
    }

    bit_lengths
}


// =========================================================================
// FSE-compressed Huffman weights encoder
// =========================================================================
//
// The literals header for a Huffman-compressed block uses one of two weight
// formats: direct (4-bit packed weights, fits up to 128 distinct symbols) or
// FSE-compressed (the weight stream is itself FSE-encoded with 2 interleaved
// states).  When the literal pool spans more than 128 distinct byte values
// the FSE-compressed form is the only option.
//
// This encoder mirrors the format produced by the C reference / ruzstd:
//   1. FSE table description (variable-length packed forward bitstream).
//   2. 2-stream interleaved FSE encoding of the weight values, written
//      so the matching decoder in `decode_weights_fse` recovers them.
// The two pieces are concatenated into a single payload whose length must
// fit in 7 bits (the literals-section header byte stores it).

/// Encode the Huffman weight stream as an FSE-compressed payload.  Returns
/// `None` on any failure (e.g. degenerate distribution, payload exceeds 127
/// bytes).  The caller should fall back to direct encoding or raw literals.
pub fn encode_weights_fse(weights: &[u8]) -> Option<Vec<u8>> {
    if weights.len() < 2 {
        return None;
    }
    // Histogram weight values 0..=11 (zstd caps Huffman weights at 11).
    let mut counts = [0u32; 12];
    let mut max_weight: usize = 0;
    for &w in weights {
        if w as usize > 11 {
            return None;
        }
        counts[w as usize] += 1;
        if w as usize > max_weight {
            max_weight = w as usize;
        }
    }
    let nonzero = counts.iter().filter(|&&c| c > 0).count();
    if nonzero < 2 {
        // FSE needs at least 2 distinct symbols.
        return None;
    }

    // Normalize counts to sum to 2^accuracy_log.
    let accuracy_log: u32 = 6;
    let table_size = 1u32 << accuracy_log;
    let norm = normalize_to_acc_log(&counts[..=max_weight], table_size as usize)?;

    // Build decoder + encoder tables from the normalized distribution.
    let dec = super::fse::FseTable::from_weights(&norm, accuracy_log).ok()?;
    let enc = super::fse::FseEncoder::from_decoder(&dec, max_weight + 1);

    // Step 1: serialize the FSE table description into a forward bitstream.
    let mut desc_bw = super::bits::ForwardBitWriter::new();
    write_fse_table_description(&mut desc_bw, &norm, accuracy_log);
    let desc_bytes = desc_bw.finalize_no_sentinel();

    // Step 2: encode the weights with 2 interleaved FSE states (the parity
    // of `n` determines the order in which the final state values are
    // written so the decoder reads them in the correct slots).
    let stream = encode_interleaved_2state(&enc, weights, accuracy_log)?;

    let mut out = Vec::with_capacity(desc_bytes.len() + stream.len());
    out.extend_from_slice(&desc_bytes);
    out.extend_from_slice(&stream);

    if out.len() >= 128 {
        // Header byte must be < 128 (literals-section header stores the size).
        return None;
    }
    Some(out)
}

/// Two-state interleaved FSE encoding of `weights`, returning the finalized
/// backward bitstream.  Mirrors ruzstd's `encode_interleaved`.
///
/// Layout (encoder time, byte stream forward):
///   * For n >= 4 the loop emits `state_1` and `state_2` transitions in
///     pairs, walking from index `n-4` down by 2 each iteration.
///   * Odd n: one extra `state_1` transition for `weights[0]`, then init
///     writes `state_2` first then `state_1`.
///   * Even n: init writes `state_1` first then `state_2`.
fn encode_interleaved_2state(
    enc: &super::fse::FseEncoder,
    weights: &[u8],
    accuracy_log: u32,
) -> Option<Vec<u8>> {
    let n = weights.len();
    if n < 2 {
        return None;
    }
    let mut bw = super::bits::ForwardBitWriter::new();

    if n == 2 {
        // No transitions — just the initial states.  Decoder reads state1
        // first, so we want state for w0 to land there.  In the even-parity
        // convention, encoder writes state_1 (state for w_{n-1}) first then
        // state_2 (state for w_{n-2}).  Decoder reads in reverse: state_2
        // (= state for w0) lands in decoder.state1.
        let s1 = enc.start_state(weights[1]); // state for w_{n-1}
        let s2 = enc.start_state(weights[0]); // state for w_{n-2}
        bw.write_bits(s1 as u64, accuracy_log);
        bw.write_bits(s2 as u64, accuracy_log);
        return Some(bw.finalize());
    }

    let mut state_1 = enc.start_state(weights[n - 1]);
    let mut state_2 = enc.start_state(weights[n - 2]);

    if n == 3 {
        // Skip the main loop and go straight to the odd-case finishing.
        state_1 = enc.encode_symbol(state_1, weights[0], &mut bw);
        // Odd parity: write state_2 first then state_1.
        bw.write_bits(state_2 as u64, accuracy_log);
        bw.write_bits(state_1 as u64, accuracy_log);
        return Some(bw.finalize());
    }

    // n >= 4: pair-wise loop.  `idx` walks the input from `n-4` down by 2.
    // Use isize so the `idx >= 0` termination check is straightforward.
    let mut idx: isize = (n as isize) - 4;
    loop {
        // state_1 transitions to weights[idx + 1]
        let target1 = weights[(idx + 1) as usize];
        state_1 = enc.encode_symbol(state_1, target1, &mut bw);
        // state_2 transitions to weights[idx]
        let target2 = weights[idx as usize];
        state_2 = enc.encode_symbol(state_2, target2, &mut bw);
        if idx < 2 {
            break;
        }
        idx -= 2;
    }

    if idx == 1 {
        // Odd n: one more state_1 transition for weights[0].
        state_1 = enc.encode_symbol(state_1, weights[0], &mut bw);
        bw.write_bits(state_2 as u64, accuracy_log);
        bw.write_bits(state_1 as u64, accuracy_log);
    } else {
        // Even n (idx == 0): no extra transition.  Init order is swapped.
        bw.write_bits(state_1 as u64, accuracy_log);
        bw.write_bits(state_2 as u64, accuracy_log);
    }

    Some(bw.finalize())
}

/// Write the FSE table description into a forward bitstream — the inverse of
/// `FseTable::decode_table` from `fse.rs`.
pub(crate) fn write_fse_table_description(
    bw: &mut super::bits::ForwardBitWriter,
    weights: &[i16],
    accuracy_log: u32,
) {
    bw.write_bits((accuracy_log - 5) as u64, 4);

    let table_size = 1u32 << accuracy_log;
    let mut remaining: i32 = table_size as i32 + 1;
    let mut threshold: i32 = table_size as i32;
    let mut nb_bits: u32 = accuracy_log + 1;

    let mut i = 0usize;
    while remaining > 1 && i < weights.len() {
        // Adjust threshold/nb_bits when remaining drops below threshold.
        while remaining < threshold && nb_bits > 1 {
            nb_bits -= 1;
            threshold >>= 1;
        }

        let prob = weights[i];
        i += 1;
        let count = (prob + 1) as i32; // 0 means "less than 1"
        let max_val = (2 * threshold - 1) - remaining;

        if count < max_val {
            // Short code: nb_bits - 1 bits.
            bw.write_bits(count as u64, nb_bits - 1);
        } else {
            // Long code: nb_bits bits, with optional adjustment.
            let value = if count >= threshold {
                count + max_val
            } else {
                count
            };
            bw.write_bits(value as u64, nb_bits);
        }

        remaining -= if prob < 0 { 1 } else { prob as i32 };

        // Repeat-zero handling: count consecutive zeros and emit in groups of 3.
        if prob == 0 {
            let mut zeros: u32 = 0;
            while i < weights.len() && weights[i] == 0 {
                zeros += 1;
                i += 1;
            }
            while zeros >= 3 {
                bw.write_bits(3, 2);
                zeros -= 3;
            }
            bw.write_bits(zeros as u64, 2);
        }
    }
}

/// Normalize raw frequency counts so they sum to exactly `target_sum`.
/// Symbols with count > 0 but rounding to 0 get the special `-1` low-prob
/// marker.  Returns None for degenerate input.
///
/// **Important:** Also caps any single probability at `target_sum / 2` so the
/// resulting FSE table has no nb=0 slots.  Without this cap a state with 0
/// transition bits lets the decoder iterate freely without consuming bits,
/// causing the 2-state interleaved Huffman-weight decoder to over-emit.
pub(crate) fn normalize_to_acc_log(counts: &[u32], target_sum: usize) -> Option<Vec<i16>> {
    let total: u32 = counts.iter().sum();
    if total == 0 {
        return None;
    }
    let max_per_symbol = (target_sum / 2) as i16;
    let mut norm: Vec<i16> = vec![0; counts.len()];
    let mut allocated: i32 = 0;
    for (i, &c) in counts.iter().enumerate() {
        if c == 0 {
            norm[i] = 0;
            continue;
        }
        let scaled = (c as u64 * target_sum as u64 + (total as u64 / 2)) / total as u64;
        if scaled == 0 {
            norm[i] = -1;
            allocated += 1;
        } else {
            let capped = (scaled as i16).min(max_per_symbol);
            norm[i] = capped;
            allocated += capped as i32;
        }
    }
    let target = target_sum as i32;
    while allocated < target {
        // Add to the largest symbol that's still under the cap.
        let max_idx = norm
            .iter()
            .enumerate()
            .filter(|(_, &v)| v > 0 && v < max_per_symbol)
            .max_by_key(|(_, &v)| v)
            .map(|(i, _)| i);
        match max_idx {
            Some(idx) => {
                norm[idx] += 1;
                allocated += 1;
            }
            None => {
                // Everything is at the cap — give up.
                return None;
            }
        }
    }
    while allocated > target {
        let max_idx = norm
            .iter()
            .enumerate()
            .filter(|(_, &v)| v > 1)
            .max_by_key(|(_, &v)| v)
            .map(|(i, _)| i)?;
        norm[max_idx] -= 1;
        allocated -= 1;
    }
    Some(norm)
}


#[cfg(test)]
mod encoder_tests {
    use super::*;

    /// Round-trip a synthetic weights vector through `encode_weights_fse` and
    /// `decode_weights_fse`.  Covers both odd and even N to exercise the
    /// parity branch in the encoder.
    ///
    /// Asserts that the decoder produces EXACTLY the same number of weights —
    /// over-/under-decoding silently changes which symbol gets the implicit
    /// last weight in the resulting Huffman table.
    fn weights_fse_roundtrip(weights: Vec<u8>) {
        let bytes = encode_weights_fse(&weights).expect("encode");
        let decoded = decode_weights_fse(&bytes).expect("decode");
        assert_eq!(
            decoded.len(),
            weights.len(),
            "decoder produced {} weights, expected {}",
            decoded.len(),
            weights.len()
        );
        for (i, &w) in weights.iter().enumerate() {
            assert_eq!(decoded[i], w, "weight {} mismatch (n={})", i, weights.len());
        }
    }

    #[test]
    fn fse_weights_roundtrip_even_n4() {
        weights_fse_roundtrip(vec![3, 2, 1, 4]);
    }

    #[test]
    fn fse_weights_roundtrip_odd_n5() {
        weights_fse_roundtrip(vec![3, 2, 1, 4, 2]);
    }

    #[test]
    fn fse_weights_roundtrip_n3() {
        weights_fse_roundtrip(vec![3, 2, 1]);
    }

    #[test]
    fn fse_weights_roundtrip_two_distinct_values() {
        // Many entries, exactly 2 distinct values.
        let mut w = Vec::new();
        for _ in 0..30 { w.push(1); }
        for _ in 0..30 { w.push(8); }
        for _ in 0..40 { w.push(0); }
        weights_fse_roundtrip(w);
    }

    #[test]
    fn fse_weights_roundtrip_synthetic_160_even() {
        let mut w: Vec<u8> = Vec::new();
        for _ in 0..30 { w.push(1); }
        for _ in 0..30 { w.push(2); }
        for _ in 0..30 { w.push(3); }
        for _ in 0..40 { w.push(0); }
        for _ in 0..30 { w.push(4); }
        assert_eq!(w.len() % 2, 0);
        weights_fse_roundtrip(w);
    }

    #[test]
    fn fse_weights_roundtrip_mostly_zeros() {
        // Mimics the real source-code distribution: ~226 weights, mostly 0,
        // a few 1s and 2s scattered throughout.
        let mut w = vec![0u8; 226];
        for i in (10..226).step_by(7) { w[i] = 1; }
        for i in (15..226).step_by(13) { w[i] = 2; }
        weights_fse_roundtrip(w);
    }

    #[test]
    fn fse_weights_roundtrip_mostly_zeros_small_n() {
        // Same shape as the real source data but at small N to make
        // regressions easy to localize.
        for n in [10, 20, 30, 50, 100, 150, 200, 226] {
            let mut w = vec![0u8; n];
            for i in (3..n).step_by(7) { w[i] = 1; }
            for i in (5..n).step_by(11) { w[i] = 2; }
            weights_fse_roundtrip(w);
        }
    }

    #[test]
    fn fse_weights_roundtrip_synthetic_161_odd() {
        let mut w: Vec<u8> = Vec::new();
        for _ in 0..30 { w.push(1); }
        for _ in 0..30 { w.push(2); }
        for _ in 0..30 { w.push(3); }
        for _ in 0..41 { w.push(0); }
        for _ in 0..30 { w.push(4); }
        assert_eq!(w.len() % 2, 1);
        weights_fse_roundtrip(w);
    }


    #[test]
    fn fse_weights_roundtrip_swept_sizes() {
        // Sweep N from 4 to 250 with a fixed distribution to exercise
        // both parities at all sizes.
        for n in 4..=250 {
            let mut w: Vec<u8> = Vec::with_capacity(n);
            for i in 0..n {
                w.push(((i * 7 + 3) % 5 + 1) as u8); // values 1..5 cycling
            }
            let bytes = match encode_weights_fse(&w) {
                Some(b) => b,
                None => continue, // some sizes may produce a degenerate distribution
            };
            let decoded = decode_weights_fse(&bytes).expect("decode");
            for (i, &expected) in w.iter().enumerate() {
                assert_eq!(
                    decoded[i], expected,
                    "n={} weight[{}] mismatch (decoded {})",
                    n, i, decoded[i]
                );
            }
        }
    }

}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_huffman_decode_text100() {
        // FSE-compressed Huffman weights from text_100 level1 zstd stream.
        let fse_weight_data = [
            0x80, 0xa9, 0x6d, 0xc0, 0x7f, 0xaf, 0x2c, 0xb6, 0x50, 0xf9, 0x71, 0xad,
            0x61, 0x52, 0x22, 0x8d, 0xc2, 0x04, 0x3e, 0x83, 0x0b, 0x33, 0xa5, 0x01,
        ];
        // Huffman bitstream (56 bytes) for decoding 99 literal bytes.
        let huf_stream = [
            0x77, 0x6e, 0x93, 0xce, 0xf4, 0xd4, 0x15, 0xe2, 0x5c, 0xfb, 0xb5, 0x6b,
            0x42, 0x33, 0x60, 0x8f, 0x88, 0xf7, 0xd4, 0x89, 0x07, 0x8b, 0xd1, 0x52,
            0x75, 0x46, 0x4f, 0x14, 0x38, 0x60, 0xb0, 0x98, 0xd1, 0x20, 0xc7, 0x3f,
            0xd6, 0xf9, 0x04, 0x5a, 0x15, 0x5a, 0x06, 0x2f, 0xb0, 0xe0, 0xa4, 0xc0,
            0x24, 0xf8, 0x60, 0xb7, 0x10, 0xfe, 0x31, 0x08,
        ];

        // Step 1: Decode weights from FSE data.
        let weights = decode_weights_fse(&fse_weight_data).unwrap();
        eprintln!("Decoded {} weights: {:?}", weights.len(), &weights);

        // Step 2: Build Huffman table.
        let table = HufTable::from_weights(&weights).unwrap();
        eprintln!("Huffman table: max_bits={}, table_size={}", table.max_bits, table.table.len());

        // Step 3: Decode literals.
        let mut literals = vec![0u8; 99];
        decode_literals_1stream_into(&table, &huf_stream, &mut literals).unwrap();
        eprintln!("Decoded {} literals, first 20: {:?}", literals.len(), &literals[..20.min(literals.len())]);
        eprintln!("As string: {:?}", std::str::from_utf8(&literals[..20.min(literals.len())]));

        let expected = b"The quick brown fox jumps over the lazy dog. Lorem ipsum dolor sit amet, consectetur adipiscing eli";
        assert_eq!(literals, expected, "literals mismatch");
    }
}
