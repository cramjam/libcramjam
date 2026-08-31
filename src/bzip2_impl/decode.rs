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
    decode_stream_streaming(input, output, None).map(|(n, _)| n)
}

/// [`decode_stream`] with optional streaming output: with a sink, every
/// finished block is written out immediately (blocks are independent, so
/// nothing needs to be kept). Returns `(input consumed, output produced)`.
pub fn decode_stream_streaming(
    input: &[u8],
    output: &mut Vec<u8>,
    mut sink: Option<&mut dyn io::Write>,
) -> io::Result<(usize, usize)> {
    let mut consumed = 0usize;
    let mut produced = 0usize;
    while consumed < input.len() {
        match decode_one_frame(&input[consumed..], output, &mut sink, &mut produced)? {
            0 => break,
            n => consumed += n,
        }
    }
    Ok((consumed, produced))
}

/// Decode exactly ONE bzip2 frame (`BZh*` header through end-of-stream
/// marker).  Returns the number of input bytes consumed.
fn decode_one_frame(
    input: &[u8],
    output: &mut Vec<u8>,
    sink: &mut Option<&mut dyn io::Write>,
    produced: &mut usize,
) -> io::Result<usize> {
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
        //
        // We populate the BWT bytes DIRECTLY into the `tt: Vec<u32>` array
        // that the inverse-BWT walker will use, with the byte in the low 8
        // bits.  We also accumulate the per-byte counts (`bucket`) inline
        // here, so the inverse-BWT setup later doesn't need a fresh count
        // pass.  Both savings together remove ~75 us / 100k from the
        // text_100k decompress profile.
        let mut tt: Vec<u32> = Vec::with_capacity(max_block_size);
        let mut bucket = [0u32; 256];
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
                    bucket[byte as usize] += zero_run;
                    let cur_len = tt.len();
                    let new_len = cur_len + zero_run as usize;
                    tt.resize(new_len, byte as u32);
                }
                break;
            }
            if sym <= 1 {
                // RUNA / RUNB — accumulate into the run length. A zero run
                // can never exceed the block size; bound it here so a
                // crafted run (RUNA/RUNB repeated) can't overflow the u32
                // accumulators or drive `tt.resize` to allocate gigabytes
                // (decompression bomb). Also stops `run_weight <<= 1` from
                // overflowing (it would after 32 doublings).
                zero_run += if sym == 0 { run_weight } else { 2 * run_weight };
                if zero_run as usize + tt.len() > max_block_size {
                    return Err(io::Error::new(
                        io::ErrorKind::InvalidData,
                        "bzip2: decoded block exceeds block size",
                    ));
                }
                run_weight <<= 1;
                continue;
            }
            // Real (non-zero) MTF index.  Flush any pending zero run first.
            if zero_run > 0 {
                let byte = alphabet_to_byte[mtf_list[0] as usize];
                bucket[byte as usize] += zero_run;
                let cur_len = tt.len();
                let new_len = cur_len + zero_run as usize;
                tt.resize(new_len, byte as u32);
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
            // A real symbol appends one byte; refuse to grow past the block
            // size (a corrupt stream could emit more symbols than fit).
            if tt.len() >= max_block_size {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidData,
                    "bzip2: decoded block exceeds block size",
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
                let byte = *alphabet_to_byte.get_unchecked(alpha_idx as usize);
                bucket[byte as usize] += 1;
                tt.push(byte as u32);
            }
        }

        // -- Fused forward inverse-BWT walk + inverse RLE1 + CRC + output extend --
        // The `tt` Vec already contains the BWT bytes in its low 8 bits and
        // `bucket` already has the per-byte counts.  The walker just needs
        // to write the FL "next" indices into the high 24 bits and walk.
        let mut block_crc = Crc32::new();
        let block_start = output.len();
        forward_inverse_bwt_rle1_crc(&mut tt, &bucket, bwt_origin, output, &mut block_crc)?;
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
        *produced += output.len() - block_start;
        if let Some(sink) = sink.as_mut() {
            sink.write_all(&output[block_start..])?;
            output.truncate(block_start);
        }
    }
}

// =========================================================================
// Fused inverse BWT + RLE1 + CRC + output write
// =========================================================================
//
// We walk the BWT in FORWARD order via the FL mapping (the inverse of the
// LF mapping a backward walk uses) and run the inverse RLE1 state machine
// inline with CRC32 update and output writes.  This is the same shape as
// libbz2's `unRLE_obuf_to_output_FAST` and avoids materialising the BWT's
// raw byte output entirely.
//
// `tt[i]` packs `(FL[i] << 8) | L[i]`:
//   - `tt[i] & 0xff` is the BWT byte at row i
//   - `tt[i] >> 8`   is the row to visit AFTER row i in cyclic forward order
//
// Walk: start at `tt[origin] >> 8`, then on each step `tt_pos = tt[tt_pos]`,
// emit `tt_pos & 0xff`, then advance with `tt_pos >>= 8`.  Bzip2 caps blocks
// at 900 KB so 24 bits of next-index is plenty.
//
// `tt` enters this function with the BWT bytes already in the low 8 bits
// (built directly during the inverse-MTF pass), and `bucket` is the per-byte
// histogram of those bytes.  Saves a pass-and-copy over `last_column`.
fn forward_inverse_bwt_rle1_crc(
    tt: &mut Vec<u32>,
    bucket: &[u32; 256],
    origin: usize,
    output: &mut Vec<u8>,
    crc: &mut Crc32,
) -> io::Result<()> {
    let n = tt.len();
    if n == 0 {
        return Ok(());
    }
    if origin >= n {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "bzip2: BWT origin out of range",
        ));
    }

    // Convert `bucket` (per-byte counts) into bucket-start offsets via a
    // running prefix sum.  These will be incremented in the FL pass below.
    let mut counts = [0u32; 256];
    let mut running = 0u32;
    for i in 0..256 {
        counts[i] = running;
        running += bucket[i];
    }

    // Build the FL "next" field of tt: walk L positions, OR each L position
    // into the high bits of tt at the next-free F slot for L[i].  The low
    // 8 bits already hold L[i] from the inverse-MTF pass.
    for i in 0..n {
        let uc = (tt[i] & 0xff) as usize;
        let f_pos = counts[uc] as usize;
        tt[f_pos] |= (i as u32) << 8;
        counts[uc] += 1;
    }

    // Forward walk + fused RLE1 + CRC + output.
    //
    // We write output bytes through a raw pointer with `set_len` called
    // ONCE at the end, skipping the per-byte capacity-check + length-update
    // overhead of `Vec::push`.  Reserve a generous upper bound on the
    // post-RLE1 expansion before entering the loop; if a pathological run
    // sequence ever pushes us past it, fall back to a slow re-reserve.
    //
    // Worst-case RLE1 expansion: any 5 bytes "bbbbN" → at most 259 output
    // bytes.  So an n-byte BWT block expands to at most ⌈n * 259 / 5⌉ output
    // bytes.  Real inputs are far below this; we cap our pre-reserve at
    // 2n + 1024 (always enough for typical data) and re-reserve on the
    // cold path if needed.
    output.reserve(n.saturating_mul(2) + 1024);
    let (mut crc_state, crc_tbl) = crc.snapshot();
    let mut t_pos = (tt[origin] >> 8) as usize;
    let mut consumed = 0usize;

    let tt_ptr = tt.as_ptr();

    // Per-iter bounds checks on `t_pos` aren't needed: the FL setup loop
    // guarantees every `tt[k] >> 8` is in `[0, n)`, so the walk can never
    // index outside the array.  Wrong input can cause the walk to enter a
    // short sub-cycle and emit garbage, but that's caught by the CRC at the
    // end.  We do a single start-of-walk check.
    if t_pos >= n {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "bzip2: BWT walk start out of range",
        ));
    }

    // Output cursor: write through `out_ptr.add(out_idx)`, then `set_len`
    // once at the end.  Bytes already in `output` (other blocks of the
    // same stream) stay intact at offsets `< out_base`.
    let out_base = output.len();
    let mut out_ptr = unsafe { output.as_mut_ptr().add(out_base) };
    let mut out_idx: usize = 0;

    // Helper macro: ensure we have at least `extra` more bytes of capacity.
    // Cold path — should never fire for typical inputs because we
    // pre-reserved 2n+1024 above.
    macro_rules! ensure_cap {
        ($extra:expr) => {{
            let need = $extra;
            if out_base + out_idx + need > output.capacity() {
                unsafe { output.set_len(out_base + out_idx); }
                output.reserve(need + 1024);
                out_ptr = unsafe { output.as_mut_ptr().add(out_base) };
            }
        }};
    }

    // Hand-unrolled inverse RLE1 state machine, mirroring bzlib's
    // `unRLE_obuf_to_output_FAST`.  The structure: the OUTER loop body
    // emits one fresh "run" of the most-recently-seen character (1 to
    // 259 copies), then prepares the next character for the following
    // iteration.  Within one iteration we read 1, 2, 3 or 5 BWT bytes
    // and statically branch on each comparison — no `run_len` variable
    // is needed because the position in the unrolled cascade IS the
    // run length so far.
    //
    // Helper macros to keep the unrolled code compact.
    macro_rules! load_next {
        () => {{
            let e = unsafe { *tt_ptr.add(t_pos) };
            let b = e as u8;
            t_pos = (e >> 8) as usize;
            consumed += 1;
            b
        }};
    }
    macro_rules! emit_n {
        ($byte:expr, $count:expr) => {{
            let bb: u8 = $byte;
            let cc: usize = $count;
            ensure_cap!(cc);
            unsafe { std::ptr::write_bytes(out_ptr.add(out_idx), bb, cc); }
            out_idx += cc;
            for _ in 0..cc {
                let idx = (((crc_state >> 24) as u8) ^ bb) as usize;
                crc_state = (crc_state << 8) ^ crc_tbl[idx];
            }
        }};
    }
    // Pre-prime: read the FIRST BWT byte into k0 (the "current run" character).
    if consumed >= n {
        unsafe { output.set_len(out_base + out_idx); }
        crc.restore(crc_state);
        return Ok(());
    }
    let mut k0 = load_next!();

    // The state-machine invariant: at the top of each iteration, exactly ONE
    // BWT character has been "buffered" in `k0` (consumed from the BWT but
    // not yet emitted).  We read up to 3 more matching characters; if all 3
    // match, the next BWT byte is the run-length count (the encoder
    // guarantees that 4 identical characters in a row are ALWAYS encoded as
    // `bbbb<count>`, so once we've seen 4 in flight we don't need a 5th
    // confirming read).
    'outer: loop {
        // Termination: the buffered k0 is the last character of the block.
        if consumed == n {
            emit_n!(k0, 1);
            break 'outer;
        }

        // Read 1st extra byte.  If it differs, we have a run-of-1 of k0.
        let k1 = load_next!();
        if k1 != k0 {
            emit_n!(k0, 1);
            k0 = k1;
            continue 'outer;
        }
        if consumed == n {
            emit_n!(k0, 2);
            break 'outer;
        }

        // Read 2nd extra byte.
        let k1 = load_next!();
        if k1 != k0 {
            emit_n!(k0, 2);
            k0 = k1;
            continue 'outer;
        }
        if consumed == n {
            emit_n!(k0, 3);
            break 'outer;
        }

        // Read 3rd extra byte.  If it matches, we have 4 chars in flight
        // (initial k0 + 3 matching reads) which the encoder ALWAYS represents
        // as a run.  Read the count next.
        let k1 = load_next!();
        if k1 != k0 {
            emit_n!(k0, 3);
            k0 = k1;
            continue 'outer;
        }
        // 4 in a row → next BWT byte is the run-length count.
        if consumed >= n {
            unsafe { output.set_len(out_base + out_idx); }
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "bzip2: RLE1 run with no count byte",
            ));
        }
        let count = load_next!() as usize;
        emit_n!(k0, 4 + count);
        // The next BWT byte is the start of the next segment.  If we're at
        // end-of-block, this run was the very last thing in the block.
        if consumed == n {
            break 'outer;
        }
        k0 = load_next!();
    }

    // Commit the new length.
    unsafe { output.set_len(out_base + out_idx); }
    crc.restore(crc_state);
    Ok(())
}

/// Test-only inverse BWT for round-tripping the encoder.  Builds the
/// `tt`/`bucket` shape that the fused walker now expects, then calls it.
#[cfg(test)]
pub(crate) fn test_only_inverse_bwt(last: &[u8], origin: usize) -> io::Result<Vec<u8>> {
    let mut tt: Vec<u32> = last.iter().map(|&b| b as u32).collect();
    let mut bucket = [0u32; 256];
    for &b in last {
        bucket[b as usize] += 1;
    }
    let mut out = Vec::new();
    let mut crc = Crc32::new();
    forward_inverse_bwt_rle1_crc(&mut tt, &bucket, origin, &mut out, &mut crc)?;
    Ok(out)
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

        // Reject over-subscribed code lengths: the canonical codes for an
        // over-full table overflow their bit width and would index past the
        // peek table. (libbzip2 tolerates *incomplete* tables via the slow
        // bit-walk, so we only reject the over-full case.)
        {
            let mut kraft: u64 = 0;
            for k in min_len..=max_len {
                kraft += (count[k as usize] as u64) << (max_len - k);
            }
            if kraft > (1u64 << max_len) {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidData,
                    "bzip2: over-subscribed Huffman code",
                ));
            }
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
                        if start + span > PEEK_TABLE_SIZE {
                            return Err(io::Error::new(
                                io::ErrorKind::InvalidData,
                                "bzip2: Huffman code exceeds peek table",
                            ));
                        }
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
