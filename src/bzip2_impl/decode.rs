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
        // Inverse MTF state: list of alphabet indices in MTF order.  Each
        // entry is a 0..num_used index into `alphabet_to_byte`, NOT a raw
        // byte — `num_used` can be 256 which doesn't fit in u8.
        let mut mtf_list: Vec<u16> = (0..num_used as u16).collect();

        let mut group_pos = 0usize;
        let mut selector_idx = 0usize;
        let mut current_table: &HufTable = &tables[selectors[0] as usize];

        let runa_b_code = 0u16; // RUNA = symbol 0, RUNB = symbol 1
        let _ = runa_b_code;

        let eob_symbol = (num_used + 1) as u16;
        let mut zero_run: u32 = 0;
        let mut run_weight: u32 = 1;

        loop {
            if group_pos == 0 {
                if selector_idx >= selectors.len() {
                    return Err(io::Error::new(
                        io::ErrorKind::InvalidData,
                        "bzip2: ran out of selectors mid-block",
                    ));
                }
                current_table = &tables[selectors[selector_idx] as usize];
                selector_idx += 1;
            }
            group_pos = (group_pos + 1) % HUFFMAN_GROUP_SIZE;

            let sym = current_table.decode(&mut br)?;
            if sym == eob_symbol {
                // Flush any pending zero run before exiting.
                flush_zero_run(&mut bwt_input, &mut mtf_list, &mut zero_run, &alphabet_to_byte)?;
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
            flush_zero_run(&mut bwt_input, &mut mtf_list, &mut zero_run, &alphabet_to_byte)?;
            run_weight = 1;

            let mtf_index = (sym - 1) as usize; // sym 2 → MTF index 1, etc.
            if mtf_index >= mtf_list.len() {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidData,
                    "bzip2: MTF index out of range",
                ));
            }
            let alpha_idx = mtf_list[mtf_index];
            // Move to front.
            for i in (1..=mtf_index).rev() {
                mtf_list[i] = mtf_list[i - 1];
            }
            mtf_list[0] = alpha_idx;
            bwt_input.push(alphabet_to_byte[alpha_idx as usize]);
        }

        // -- Inverse BWT --
        let plain = inverse_bwt(&bwt_input, bwt_origin)?;

        // -- Inverse RLE1 --
        let block_bytes = inverse_rle1(&plain);

        // -- CRC --
        let mut block_crc = Crc32::new();
        block_crc.update(&block_bytes);
        let computed_block_crc = block_crc.finalize();
        if computed_block_crc != stored_block_crc {
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

        output.extend_from_slice(&block_bytes);
    }
}

fn flush_zero_run(
    out: &mut Vec<u8>,
    mtf_list: &mut Vec<u16>,
    zero_run: &mut u32,
    alphabet_to_byte: &[u8; 256],
) -> io::Result<()> {
    let n = *zero_run as usize;
    if n == 0 {
        return Ok(());
    }
    *zero_run = 0;
    // The MTF index 0 corresponds to mtf_list[0].  We don't move-to-front
    // for index 0 (it's already at the front).  Just emit n copies.
    if mtf_list.is_empty() {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "bzip2: zero-run with empty MTF list",
        ));
    }
    let byte_alpha = mtf_list[0];
    let byte = alphabet_to_byte[byte_alpha as usize];
    for _ in 0..n {
        out.push(byte);
    }
    Ok(())
}

// =========================================================================
// Inverse Burrows-Wheeler Transform
// =========================================================================
//
// Standard counting-sort + cycle-following algorithm:
//   1. Compute P[i] for each i where P[i] is the row of L[i] in the sorted
//      first-column ordering, broken by ascending original index.
//   2. Starting from `origin`, walk P for N steps, emitting L[walk] in
//      REVERSE output order (so the first emit fills the last output slot).
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

    // P[i] = (count of L[j] < L[i] for any j) + (count of L[j] == L[i] for j < i)
    let mut p = vec![0u32; len];
    for (i, &b) in last_column.iter().enumerate() {
        let s = counts[b as usize];
        p[i] = s;
        counts[b as usize] = s + 1;
    }

    // Walk N steps from origin, filling the output BACKWARD.
    let mut out = vec![0u8; len];
    let mut j = origin;
    for i in (0..len).rev() {
        out[i] = last_column[j];
        j = p[j] as usize;
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
fn inverse_rle1(input: &[u8]) -> Vec<u8> {
    let mut out: Vec<u8> = Vec::with_capacity(input.len());
    let mut i = 0usize;
    let mut run_len = 0usize;
    let mut last: u8 = 0;
    while i < input.len() {
        let b = input[i];
        i += 1;
        if run_len > 0 && b == last {
            run_len += 1;
            out.push(b);
            if run_len == 4 {
                // Next byte is the extra-run-length count.
                if i < input.len() {
                    let extra = input[i] as usize;
                    i += 1;
                    for _ in 0..extra {
                        out.push(b);
                    }
                }
                // Reset the run counter so the next byte starts fresh.
                run_len = 0;
            }
        } else {
            last = b;
            run_len = 1;
            out.push(b);
        }
    }
    out
}

// =========================================================================
// Canonical Huffman decoder for bzip2's per-table code-length lists.
// =========================================================================

struct HufTable {
    // base[k] = (first_code_at_length_k << (max_len - k)) computed once;
    // limit[k] is the largest code (left-justified) of length k.
    // For each length, codes are assigned in canonical order with lower
    // symbol indexes getting smaller codes.
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
        // Assign canonical codes: at length min_len, first code = 0; at
        // each subsequent length, the next code is `(prev_max + 1) << 1`.
        let mut code: u32 = 0;
        let mut idx = 0u32;
        for k in min_len..=max_len {
            let n = count[k as usize];
            // Code range for length k: [code .. code + n).
            // limit[k] = (code + n - 1) left-justified to max_len.
            // base[k] = code - idx (so symbol_offset = (left_justified_code >> (max_len - k)) - base[k] gives perm index).
            let last = code + n - 1;
            limit[k as usize] = last << (max_len - k);
            base[k as usize] = code as i32 - idx as i32;
            idx += n;
            code = (last + 1) << 1;
        }

        Ok(Self {
            min_len,
            max_len,
            limit,
            base,
            perm,
        })
    }

    /// Decode one Huffman symbol from the bit stream.  Reads bits one at a
    /// time until a length matches `limit[k]`.
    #[inline]
    fn decode(&self, br: &mut BitReader<'_>) -> io::Result<u16> {
        // Start by reading min_len bits and left-justifying to max_len.
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
            // Read one more bit and OR it into the appropriate position.
            let next = br.read_bits(1)?;
            code |= next << (self.max_len - k - 1);
            k += 1;
        }
    }
}
