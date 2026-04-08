//! bzip2 decoder.
//!
//! Pipeline (per block, in reverse order from how the encoder ran):
//!   1. Parse the block header (CRC, BWT origin, used-symbol bitmap).
//!   2. Read the Huffman tables and selector list.
//!   3. Decode the post-Huffman symbol stream (RUNA/RUNB + MTF indices + EOB).
//!   4. Reverse RLE2 (RUNA/RUNB → zero runs in the MTF index stream).
//!   5. Inverse MTF → recovers the BWT output bytes.
//!   6. Inverse BWT → recovers the RLE1-pre-processed bytes.
//!   7. Inverse RLE1 → recovers the original bytes.
//!   8. Verify the per-block CRC against the recovered bytes.
//!
//! At the file level we also verify the combined CRC at the end-of-stream
//! marker, and we transparently decode multi-stream files (multiple
//! "BZh*" headers concatenated, as produced by `bzip2 -c file1 file2 > out`).

use std::io;

use super::bits::BitReader;
use super::crc::Crc32;

const FILE_MAGIC_HUFFMAN: u8 = b'h'; // bzip2 always uses Huffman = 'h'
const BLOCK_MAGIC: u64 = 0x3141_5926_5359; // 48 bits
const EOS_MAGIC: u64 = 0x1772_4538_5090; // 48 bits

const MAX_ALPHA: usize = 258; // RUNA + RUNB + 0..255 MTF indices + EOB
const MAX_HUFFMAN_TABLES: usize = 6;
const MAX_HUFFMAN_CODE_LEN: usize = 23; // bzip2 limit
const HUFFMAN_GROUP_SIZE: usize = 50;

/// Decode an entire bzip2 stream (one or more concatenated frames) into
/// `output`, returning the total number of input bytes consumed.
pub fn decode_stream(input: &[u8], output: &mut Vec<u8>) -> io::Result<usize> {
    let mut consumed = 0usize;
    while consumed < input.len() {
        match decode_one_frame(&input[consumed..], output)? {
            0 => break,
            n => consumed += n,
        }
    }
    Ok(consumed)
}

/// Decode exactly ONE bzip2 frame (`BZh*` header through end-of-stream
/// marker).  Returns the number of input bytes consumed.
fn decode_one_frame(input: &[u8], output: &mut Vec<u8>) -> io::Result<usize> {
    if input.len() < 4 {
        return Ok(0);
    }
    if input[0] != b'B' || input[1] != b'Z' || input[2] != FILE_MAGIC_HUFFMAN {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "bzip2: bad file magic",
        ));
    }
    let level = input[3];
    if !(b'1'..=b'9').contains(&level) {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "bzip2: bad level digit",
        ));
    }
    let max_block_size = (level - b'0') as usize * 100_000;

    let mut br = BitReader::new(&input[4..]);
    let mut combined_crc: u32 = 0;

    loop {
        // Each iteration starts with a 48-bit magic number that distinguishes
        // a regular block from the end-of-stream marker.
        let magic = br.read_bits_u64(48)?;
        if magic == EOS_MAGIC {
            // 32-bit combined CRC follows; we verify against our running
            // combined CRC.
            let stored_combined = br.read_bits(32)?;
            if stored_combined != combined_crc {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidData,
                    format!(
                        "bzip2: combined CRC mismatch (stored {:08x}, computed {:08x})",
                        stored_combined, combined_crc
                    ),
                ));
            }
            br.align_to_byte();
            // Bytes consumed = 4-byte file header + (bits_consumed_since_header / 8).
            let bytes_after_header = (br.bit_position() + 7) / 8;
            return Ok(4 + bytes_after_header);
        }
        if magic != BLOCK_MAGIC {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!("bzip2: bad block/EOS magic 0x{:012x}", magic),
            ));
        }

        // -- Block header --
        let stored_block_crc = br.read_bits(32)?;
        let randomized = br.read_bit()?;
        if randomized {
            // The "randomized" mode hasn't been used by bzip2 since the late
            // 1990s.  Reject it loudly so we don't silently mis-decode.
            return Err(io::Error::new(
                io::ErrorKind::Unsupported,
                "bzip2: randomized blocks not supported",
            ));
        }
        let bwt_origin = br.read_bits(24)? as usize;

        // Used-symbol bitmap: 16-bit "high" map + 16-bit "low" map per set
        // bit in the high map.
        let mut symbol_used = [false; 256];
        let high = br.read_bits(16)?;
        for hi in 0..16 {
            if (high >> (15 - hi)) & 1 != 0 {
                let low = br.read_bits(16)?;
                for lo in 0..16 {
                    if (low >> (15 - lo)) & 1 != 0 {
                        symbol_used[hi * 16 + lo] = true;
                    }
                }
            }
        }
        let num_used: usize = symbol_used.iter().filter(|&&u| u).count();
        if num_used == 0 {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "bzip2: empty symbol map",
            ));
        }

        // Build the inverse map: alphabet symbol → original byte.
        let mut alphabet_to_byte = [0u8; 256];
        {
            let mut idx = 0usize;
            for (b, used) in symbol_used.iter().enumerate() {
                if *used {
                    alphabet_to_byte[idx] = b as u8;
                    idx += 1;
                }
            }
        }

        // The Huffman alphabet is num_used + 2 symbols:
        //   0     = RUNA
        //   1     = RUNB
        //   2..N  = MTF indices 1..N-1
        //   N+1   = EOB     (where N = num_used + 1, so EOB = num_used + 1)
        // Wait — alphabet size is num_used + 2; symbols 2..num_used cover
        // MTF indices 1..num_used-1, and symbol num_used+1 = EOB.
        let alpha_size = num_used + 2;
        debug_assert!(alpha_size <= MAX_ALPHA);

        // -- Huffman tables --
        let num_tables = br.read_bits(3)? as usize;
        if !(2..=MAX_HUFFMAN_TABLES).contains(&num_tables) {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "bzip2: invalid Huffman table count",
            ));
        }
        let num_selectors = br.read_bits(15)? as usize;
        if num_selectors == 0 || num_selectors > 18002 {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "bzip2: invalid selector count",
            ));
        }

        // Selector list: each selector is a unary-encoded MTF index into a
        // table list of length `num_tables`.  We then inverse-MTF that list
        // to get the actual table indices.
        let mut selectors_mtf: Vec<u8> = Vec::with_capacity(num_selectors);
        for _ in 0..num_selectors {
            let mut k = 0u8;
            while br.read_bit()? {
                k += 1;
                if k as usize >= num_tables {
                    return Err(io::Error::new(
                        io::ErrorKind::InvalidData,
                        "bzip2: selector MTF index out of range",
                    ));
                }
            }
            selectors_mtf.push(k);
        }
        // MTF-decode selectors.
        let mut pos: [u8; MAX_HUFFMAN_TABLES] = [0, 1, 2, 3, 4, 5];
        let mut selectors: Vec<u8> = Vec::with_capacity(num_selectors);
        for &k in &selectors_mtf {
            let k = k as usize;
            let v = pos[k];
            // Move-to-front.
            for i in (1..=k).rev() {
                pos[i] = pos[i - 1];
            }
            pos[0] = v;
            selectors.push(v);
        }

        // Per-table code lengths: 5-bit initial length, then a "delta" for
        // each symbol (1 bit "more": loop reading 1 bit per "more"; each
        // additional bit is +1 if 0 / -1 if 1).
        let mut code_lens: Vec<Vec<u8>> = vec![vec![0; alpha_size]; num_tables];
        for t in 0..num_tables {
            let mut current = br.read_bits(5)? as i32;
            for s in 0..alpha_size {
                loop {
                    if !(1..=20).contains(&current) {
                        return Err(io::Error::new(
                            io::ErrorKind::InvalidData,
                            "bzip2: code length out of range",
                        ));
                    }
                    if !br.read_bit()? {
                        break;
                    }
                    if br.read_bit()? {
                        current -= 1;
                    } else {
                        current += 1;
                    }
                }
                code_lens[t][s] = current as u8;
            }
        }

        // Build per-table decode tables.
        let mut tables: Vec<HufTable> = Vec::with_capacity(num_tables);
        for lens in &code_lens {
            tables.push(HufTable::from_lengths(lens)?);
        }

        // -- Decode the post-Huffman symbol stream into MTF indices --
        // The output goes through inverse RLE2 (RUNA/RUNB → zero runs)
        // then inverse MTF.  We do RLE2 inline; MTF inverse is below.
        let mut bwt_input: Vec<u8> = Vec::with_capacity(max_block_size);
        // Inverse MTF state.  Fixed-size [u16; 256] arrays let LLVM elide
        // bounds checks and keep everything in L1.  `mtf_list[i]` is the
        // alphabet-index (into `alphabet_to_byte`) currently at MTF position i.
        let mut mtf_list: [u16; 256] = [0; 256];
        for i in 0..num_used {
            mtf_list[i] = i as u16;
        }

        let mut group_left: usize = 0;
        let mut selector_idx = 0usize;
        let mut current_table: &HufTable = &tables[selectors[0] as usize];

        let eob_symbol = (num_used + 1) as u16;
        let mut zero_run: u32 = 0;
        let mut run_weight: u32 = 1;
        let max_mtf_index = num_used; // valid range 1..=num_used (sym 2..=num_used+1)

        loop {
            if group_left == 0 {
                if selector_idx >= selectors.len() {
                    return Err(io::Error::new(
                        io::ErrorKind::InvalidData,
                        "bzip2: ran out of selectors mid-block",
                    ));
                }
                current_table = &tables[selectors[selector_idx] as usize];
                selector_idx += 1;
                group_left = HUFFMAN_GROUP_SIZE;
            }
            group_left -= 1;

            let sym = current_table.decode(&mut br)?;
            if sym == eob_symbol {
                // Flush any pending zero run before exiting.
                if zero_run > 0 {
                    let byte = alphabet_to_byte[mtf_list[0] as usize];
                    let new_len = bwt_input.len() + zero_run as usize;
                    bwt_input.resize(new_len, byte);
                }
                break;
            }
            if sym <= 1 {
                // RUNA / RUNB — accumulate into the run length.
                if sym == 0 {
                    zero_run += run_weight;
                } else {
                    zero_run += 2 * run_weight;
                }
                run_weight <<= 1;
                continue;
            }
            // Real (non-zero) MTF index.  Flush any pending zero run first.
            if zero_run > 0 {
                let byte = alphabet_to_byte[mtf_list[0] as usize];
                let new_len = bwt_input.len() + zero_run as usize;
                bwt_input.resize(new_len, byte);
                zero_run = 0;
            }
            run_weight = 1;

            let mtf_index = (sym - 1) as usize; // sym 2 → MTF index 1, etc.
            if mtf_index >= max_mtf_index {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidData,
                    "bzip2: MTF index out of range",
                ));
            }
            // Move-to-front: shift mtf_list[0..mtf_index] right by one slot
            // (memmove via copy_within), then update slot 0.  No position
            // table needed because the decoder receives the index directly.
            unsafe {
                let p = mtf_list.as_mut_ptr();
                let alpha_idx = *p.add(mtf_index);
                std::ptr::copy(p, p.add(1), mtf_index);
                *p = alpha_idx;
                bwt_input.push(*alphabet_to_byte.get_unchecked(alpha_idx as usize));
            }
        }

        // -- Inverse BWT --
        let plain = inverse_bwt(&bwt_input, bwt_origin)?;

        // -- Fused inverse RLE1 + CRC + output extend --
        // Instead of building an intermediate `block_bytes` vec, expand the
        // RLE1 stream directly into `output` while updating the per-block
        // CRC.  Saves one allocation and one pass over the data.
        let mut block_crc = Crc32::new();
        let block_start = output.len();
        inverse_rle1_into(&plain, output, &mut block_crc);
        let computed_block_crc = block_crc.finalize();
        if computed_block_crc != stored_block_crc {
            // Roll back the partial block we just wrote so the caller doesn't
            // see corrupted data on the error path.
            output.truncate(block_start);
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!(
                    "bzip2: block CRC mismatch (stored {:08x}, computed {:08x})",
                    stored_block_crc, computed_block_crc
                ),
            ));
        }
        // Combined CRC: rotate left by 1 and XOR (per the spec).
        combined_crc = combined_crc.rotate_left(1) ^ stored_block_crc;
    }
}

// =========================================================================
// Inverse Burrows-Wheeler Transform
// =========================================================================
//
// Standard counting-sort + cycle-following algorithm.  We pack the
// "next-index" and the "byte" together into a single u32 entry per BWT
// position so the hot walk is a single random-access load per step instead
// of two — halves the cache misses on the random-access walk, which is the
// main bottleneck.  Bzip2 limits blocks to 900 KB so 24 bits of next-index
// is plenty.
fn inverse_bwt(last_column: &[u8], origin: usize) -> io::Result<Vec<u8>> {
    let len = last_column.len();
    if len == 0 {
        return Ok(Vec::new());
    }
    if origin >= len {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "bzip2: BWT origin out of range",
        ));
    }

    // Counting sort: counts[k] becomes the index in the sorted column where
    // occurrences of byte k begin (after the prefix-sum pass).
    let mut counts = [0u32; 256];
    for &b in last_column {
        counts[b as usize] += 1;
    }
    let mut running = 0u32;
    for c in counts.iter_mut() {
        let n = *c;
        *c = running;
        running += n;
    }

    // p_byte[i] = ((next_index << 8) | byte) for BWT row i.
    let mut p_byte = vec![0u32; len];
    for (i, &b) in last_column.iter().enumerate() {
        let s = counts[b as usize];
        p_byte[i] = (s << 8) | b as u32;
        counts[b as usize] = s + 1;
    }

    // Walk N steps from `origin`, filling the output BACKWARD.  Single
    // random-access load per step.  We issue a software prefetch one step
    // ahead so the next entry can be in flight while we process the current
    // one — this masks part of the L2/L3 latency on the random-access walk.
    let mut out = vec![0u8; len];
    let mut j = origin;
    unsafe {
        let pb = p_byte.as_ptr();
        let dst = out.as_mut_ptr();
        for i in (0..len).rev() {
            let entry = *pb.add(j);
            let j_next = (entry >> 8) as usize;
            #[cfg(target_arch = "x86_64")]
            {
                use std::arch::x86_64::{_mm_prefetch, _MM_HINT_T0};
                _mm_prefetch(pb.add(j_next) as *const i8, _MM_HINT_T0);
            }
            #[cfg(target_arch = "aarch64")]
            {
                use std::arch::aarch64::{_prefetch, _PREFETCH_LOCALITY3, _PREFETCH_READ};
                _prefetch(pb.add(j_next) as *const i8, _PREFETCH_READ, _PREFETCH_LOCALITY3);
            }
            *dst.add(i) = entry as u8;
            j = j_next;
        }
    }
    Ok(out)
}

/// Test-only re-export of `inverse_bwt` for cross-checking the encoder.
#[cfg(test)]
pub(crate) fn test_only_inverse_bwt(last: &[u8], origin: usize) -> io::Result<Vec<u8>> {
    inverse_bwt(last, origin)
}

// =========================================================================
// Inverse RLE1
// =========================================================================
//
// Forward RLE1: any run of 4..=255 of the same byte is encoded as the byte
// repeated 4 times followed by 1 byte giving (run_length - 4).  Long runs
// are split across multiple groups (a group has at most 4 + 255 = 259
// bytes; for `n > 4 + 255`, the encoder emits another `bbbb<count>`).
//
// Inverse: scan the INPUT counting consecutive identical bytes.  When the
// run reaches 4, the next input byte is the count and we expand by that
// many copies.  After expansion, the counter resets — even if the next
// input byte is again the same, it doesn't add to a brand-new run yet.
fn inverse_rle1_into(input: &[u8], out: &mut Vec<u8>, crc: &mut Crc32) {
    // Reserve a generous lower bound to keep this from realloc-ing in the
    // common case.  RLE1 expansion is at most ~64x for pathological runs but
    // is usually <2x; this just avoids the first few re-grows.
    out.reserve(input.len());
    let (mut state, tbl) = crc.snapshot();
    let mut i = 0usize;
    let mut run_len = 0usize;
    let mut last: u8 = 0;
    while i < input.len() {
        let b = input[i];
        i += 1;
        if run_len > 0 && b == last {
            run_len += 1;
            out.push(b);
            // Inline CRC update — tbl reference is already in scope, no
            // OnceLock get per byte.
            let idx = (((state >> 24) as u8) ^ b) as usize;
            state = (state << 8) ^ tbl[idx];
            if run_len == 4 {
                // Next byte is the extra-run-length count.
                if i < input.len() {
                    let extra = input[i] as usize;
                    i += 1;
                    if extra > 0 {
                        // Bulk-extend then bulk-CRC the run.
                        let start = out.len();
                        out.resize(start + extra, b);
                        // CRC the run inline.
                        for _ in 0..extra {
                            let idx = (((state >> 24) as u8) ^ b) as usize;
                            state = (state << 8) ^ tbl[idx];
                        }
                    }
                }
                run_len = 0;
            }
        } else {
            last = b;
            run_len = 1;
            out.push(b);
            let idx = (((state >> 24) as u8) ^ b) as usize;
            state = (state << 8) ^ tbl[idx];
        }
    }
    crc.restore(state);
}

// =========================================================================
// Canonical Huffman decoder for bzip2's per-table code-length lists.
// =========================================================================
//
// The hot decode path uses a `PEEK_LEN`-bit lookup table: peek the top
// `PEEK_LEN` bits of the bit stream, look up `(symbol, code_len)`, advance
// by `code_len` bits.  Codes longer than `PEEK_LEN` (rare for small alpha
// sizes) fall through to a per-bit walk.

const PEEK_LEN: u32 = 10;
const PEEK_TABLE_SIZE: usize = 1 << PEEK_LEN;

struct HufTable {
    min_len: u8,
    max_len: u8,
    /// `limit[k]` = max canonical code for length k, left-justified to max_len.
    limit: [u32; MAX_HUFFMAN_CODE_LEN + 1],
    /// `base[k]` = `first_code[k] - count_through_k` for the canonical
    /// algorithm.  Used to compute the symbol index after a length match.
    base: [i32; MAX_HUFFMAN_CODE_LEN + 1],
    /// `perm[i]` = symbol assigned to the i-th canonical code (in length
    /// order, lower symbols first).
    perm: Vec<u16>,
    /// Per-pattern (sym, len) lookup keyed by the top `PEEK_LEN` bits of the
    /// bit stream.  `peek_len[i] == 0` indicates a code longer than
    /// `PEEK_LEN` — fall through to the slow path.
    peek_sym: Box<[u16; PEEK_TABLE_SIZE]>,
    peek_len: Box<[u8; PEEK_TABLE_SIZE]>,
}

impl HufTable {
    fn from_lengths(lens: &[u8]) -> io::Result<Self> {
        if lens.is_empty() {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "bzip2: empty Huffman length list",
            ));
        }
        let mut min_len = u8::MAX;
        let mut max_len = 0u8;
        for &l in lens {
            if l == 0 || l as usize > MAX_HUFFMAN_CODE_LEN {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidData,
                    "bzip2: Huffman length out of range",
                ));
            }
            if l < min_len {
                min_len = l;
            }
            if l > max_len {
                max_len = l;
            }
        }

        let mut limit = [0u32; MAX_HUFFMAN_CODE_LEN + 1];
        let mut base = [0i32; MAX_HUFFMAN_CODE_LEN + 1];

        // Count symbols per length.
        let mut count = [0u32; MAX_HUFFMAN_CODE_LEN + 2];
        for &l in lens {
            count[l as usize] += 1;
        }

        // Build perm[] in canonical order (sorted by length asc, symbol asc).
        let mut perm: Vec<u16> = Vec::with_capacity(lens.len());
        for length in min_len..=max_len {
            for (sym, &l) in lens.iter().enumerate() {
                if l == length {
                    perm.push(sym as u16);
                }
            }
        }

        // Compute limit[k] and base[k] using the standard algorithm.
        let mut code: u32 = 0;
        let mut idx = 0u32;
        for k in min_len..=max_len {
            let n = count[k as usize];
            let last = code + n - 1;
            limit[k as usize] = last << (max_len - k);
            base[k as usize] = code as i32 - idx as i32;
            idx += n;
            code = (last + 1) << 1;
        }

        // Build the peek table.  Walk perm in code-order, computing the
        // canonical code for each symbol; for codes ≤ PEEK_LEN, fan out
        // across `1 << (PEEK_LEN - k)` slots in the peek table.  For longer
        // codes we leave peek_len[slot] = 0 so the decoder falls back to the
        // bit-walk.
        let mut peek_sym: Box<[u16; PEEK_TABLE_SIZE]> =
            vec![0u16; PEEK_TABLE_SIZE].into_boxed_slice().try_into().unwrap();
        let mut peek_len: Box<[u8; PEEK_TABLE_SIZE]> =
            vec![0u8; PEEK_TABLE_SIZE].into_boxed_slice().try_into().unwrap();
        {
            let mut code: u32 = 0;
            let mut perm_idx = 0usize;
            for k in 1..=max_len as u32 {
                let n = count[k as usize] as usize;
                if k <= PEEK_LEN {
                    let pad = PEEK_LEN - k;
                    let span = 1usize << pad;
                    for _ in 0..n {
                        let start = (code as usize) << pad;
                        let sym = perm[perm_idx];
                        for slot in start..start + span {
                            peek_sym[slot] = sym;
                            peek_len[slot] = k as u8;
                        }
                        code += 1;
                        perm_idx += 1;
                    }
                } else {
                    // Skip past these codes (leave peek_len = 0 for their
                    // prefix slots so the decoder takes the slow path).
                    code += n as u32;
                    perm_idx += n;
                }
                code <<= 1;
            }
        }

        Ok(Self {
            min_len,
            max_len,
            limit,
            base,
            perm,
            peek_sym,
            peek_len,
        })
    }

    /// Decode one Huffman symbol from the bit stream.  Hot path: peek
    /// `PEEK_LEN` bits and look up `(sym, code_len)` in O(1).  Slow path:
    /// per-bit walk for codes longer than `PEEK_LEN`.
    #[inline]
    fn decode(&self, br: &mut BitReader<'_>) -> io::Result<u16> {
        // Refill so we have at least PEEK_LEN bits ready.  If the input
        // is too short for PEEK_LEN, fall through to the slow path which
        // handles short tails one bit at a time.
        if br.refill(PEEK_LEN).is_ok() {
            let p = br.peek(PEEK_LEN) as usize;
            let len = self.peek_len[p];
            if len > 0 {
                br.consume(len as u32);
                return Ok(self.peek_sym[p]);
            }
        }
        self.decode_slow(br)
    }

    /// Slow per-bit decode used for codes longer than `PEEK_LEN` and at the
    /// very end of the bitstream when `refill(PEEK_LEN)` fails.
    #[cold]
    fn decode_slow(&self, br: &mut BitReader<'_>) -> io::Result<u16> {
        let mut k = self.min_len;
        let mut code: u32 = br.read_bits(k as u32)? << (self.max_len - k);
        loop {
            if code <= self.limit[k as usize] {
                let perm_idx = ((code >> (self.max_len - k)) as i32 - self.base[k as usize]) as usize;
                if perm_idx >= self.perm.len() {
                    return Err(io::Error::new(
                        io::ErrorKind::InvalidData,
                        "bzip2: Huffman decode index out of range",
                    ));
                }
                return Ok(self.perm[perm_idx]);
            }
            if k == self.max_len {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidData,
                    "bzip2: Huffman code longer than max_len",
                ));
            }
            let next = br.read_bits(1)?;
            code |= next << (self.max_len - k - 1);
            k += 1;
        }
    }
}
