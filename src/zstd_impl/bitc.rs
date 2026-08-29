//! C-style forward bit stream (`BIT_CStream_t`) and the FSE compression
//! table (`FSE_CTable`) used by the sequence and Huffman-weight encoders.
//!
//! Bits accumulate LSB-first in a 64-bit container and are flushed 8 bytes
//! at a time into a pre-reserved output buffer, so the hot path has no
//! capacity checks.

/// Forward bit writer over the tail of a `Vec<u8>`.
///
/// The caller passes the maximum number of bytes it will write; `new`
/// reserves that plus 8 bytes of slack for the unconditional 8-byte flush.
pub struct BitCStream<'a> {
    out: &'a mut Vec<u8>,
    start: usize,
    pos: usize,
    container: u64,
    bit_pos: u32,
}

impl<'a> BitCStream<'a> {
    pub fn new(out: &'a mut Vec<u8>, max_bytes: usize) -> Self {
        out.reserve(max_bytes + 16);
        let start = out.len();
        BitCStream { out, start, pos: start, container: 0, bit_pos: 0 }
    }

    /// Append the low `nb` bits of `value` (`nb <= 32`; the caller keeps
    /// `bit_pos + nb <= 64` by flushing).
    #[inline(always)]
    pub fn add_bits(&mut self, value: u32, nb: u32) {
        debug_assert!(self.bit_pos + nb <= 64);
        let mask = (1u64 << nb) - 1;
        self.container |= (value as u64 & mask) << self.bit_pos;
        self.bit_pos += nb;
    }

    /// Flush the whole bytes of the container with one unaligned 8-byte
    /// store (up to 7 bits stay).
    #[inline(always)]
    pub fn flush(&mut self) {
        let nb_bytes = (self.bit_pos >> 3) as usize;
        debug_assert!(self.pos + 8 <= self.out.capacity());
        unsafe {
            core::ptr::write_unaligned(self.out.as_mut_ptr().add(self.pos) as *mut u64, self.container.to_le());
        }
        self.pos += nb_bytes;
        self.bit_pos &= 7;
        self.container >>= nb_bytes * 8;
    }

    /// Add the end-of-stream sentinel bit, flush, and commit the length.
    /// Returns the number of bytes written since `new`.
    pub fn close(mut self) -> usize {
        self.add_bits(1, 1);
        self.flush();
        let total = self.pos + (self.bit_pos > 0) as usize;
        debug_assert!(total <= self.out.capacity());
        unsafe { self.out.set_len(total) };
        total - self.start
    }

    /// Flush and commit without a sentinel (forward-packed streams such as
    /// FSE table descriptions).
    pub fn close_no_sentinel(mut self) -> usize {
        self.flush();
        let total = self.pos + (self.bit_pos > 0) as usize;
        debug_assert!(total <= self.out.capacity());
        unsafe { self.out.set_len(total) };
        total - self.start
    }
}

// ---------------------------------------------------------------------------
// FSE compression table (FSE_buildCTable)
// ---------------------------------------------------------------------------

#[derive(Clone, Copy, Default)]
struct SymbolTT {
    delta_find_state: i32,
    delta_nb_bits: u32,
}

/// `FSE_CTable`: `state_table` (next state per sorted slot) and one
/// `SymbolTT` per symbol.
pub struct FseCTable {
    pub table_log: u32,
    state_table: Vec<u16>,
    symbol_tt: Vec<SymbolTT>,
}

/// Encoder state (`FSE_CState_t`).
#[derive(Clone, Copy)]
pub struct FseCState {
    value: u32,
}

impl FseCTable {
    /// Build from a normalized distribution (`norm.iter().map(|&n| if n == -1 {1} else {n}).sum() == 1 << table_log`).
    pub fn build(norm: &[i16], table_log: u32) -> Self {
        let table_size = 1usize << table_log;
        let table_mask = table_size - 1;
        let step = (table_size >> 1) + (table_size >> 3) + 3;
        let max_sv1 = norm.len();
        let mut cumul = vec![0u16; max_sv1 + 1];
        let mut table_symbol = vec![0u8; table_size];
        let mut high_threshold = table_size - 1;

        for u in 1..=max_sv1 {
            if norm[u - 1] == -1 {
                cumul[u] = cumul[u - 1] + 1;
                table_symbol[high_threshold] = (u - 1) as u8;
                high_threshold -= 1;
            } else {
                cumul[u] = cumul[u - 1] + norm[u - 1] as u16;
            }
        }
        cumul[max_sv1] = (table_size + 1) as u16;

        // Spread symbols.
        let mut position = 0usize;
        for (symbol, &freq) in norm.iter().enumerate() {
            for _ in 0..freq.max(0) {
                table_symbol[position] = symbol as u8;
                position = (position + step) & table_mask;
                while position > high_threshold {
                    position = (position + step) & table_mask;
                }
            }
        }
        debug_assert_eq!(position, 0);

        // Build the state table.
        let mut state_table = vec![0u16; table_size];
        for (u, &s) in table_symbol.iter().enumerate() {
            let c = &mut cumul[s as usize];
            state_table[*c as usize] = (table_size + u) as u16;
            *c += 1;
        }

        // Symbol transformation table.
        let mut symbol_tt = vec![SymbolTT::default(); max_sv1];
        let mut total: u32 = 0;
        for (s, &n) in norm.iter().enumerate() {
            match n {
                0 => {
                    symbol_tt[s].delta_nb_bits = ((table_log + 1) << 16) - (1 << table_log);
                }
                -1 | 1 => {
                    symbol_tt[s].delta_nb_bits = (table_log << 16) - (1 << table_log);
                    symbol_tt[s].delta_find_state = total as i32 - 1;
                    total += 1;
                }
                _ => {
                    let n = n as u32;
                    let max_bits_out = table_log - super::seqstore::highbit32(n - 1);
                    let min_state_plus = n << max_bits_out;
                    symbol_tt[s].delta_nb_bits = (max_bits_out << 16) - min_state_plus;
                    symbol_tt[s].delta_find_state = total as i32 - n as i32;
                    total += n;
                }
            }
        }
        FseCTable { table_log, state_table, symbol_tt }
    }

    /// `FSE_buildCTable_rle`: a single symbol, zero bits per symbol.
    pub fn rle(symbol: u8) -> Self {
        let mut symbol_tt = vec![SymbolTT::default(); symbol as usize + 1];
        symbol_tt[symbol as usize] = SymbolTT { delta_find_state: 0, delta_nb_bits: 0 };
        FseCTable { table_log: 0, state_table: vec![0u16], symbol_tt }
    }

    /// `FSE_initCState2`: first state for `symbol` (the last symbol encoded).
    #[inline(always)]
    pub fn init_state(&self, symbol: u8) -> FseCState {
        let tt = self.symbol_tt[symbol as usize];
        let nb_bits_out = (tt.delta_nb_bits.wrapping_add(1 << 15)) >> 16;
        let value = (nb_bits_out << 16).wrapping_sub(tt.delta_nb_bits);
        let idx = ((value >> nb_bits_out) as i32 + tt.delta_find_state) as usize;
        FseCState { value: self.state_table[idx] as u32 }
    }

    /// `FSE_encodeSymbol`.
    ///
    /// # Safety
    /// `symbol` must be within the table's alphabet and have non-zero
    /// probability.
    #[inline(always)]
    pub unsafe fn encode(&self, state: &mut FseCState, symbol: u8, bits: &mut BitCStream) {
        let tt = unsafe { *self.symbol_tt.get_unchecked(symbol as usize) };
        let nb_bits_out = state.value.wrapping_add(tt.delta_nb_bits) >> 16;
        bits.add_bits(state.value, nb_bits_out);
        let idx = ((state.value >> nb_bits_out) as i32 + tt.delta_find_state) as usize;
        state.value = unsafe { *self.state_table.get_unchecked(idx) } as u32;
    }

    /// `FSE_flushCState`.
    #[inline(always)]
    pub fn flush_state(&self, state: &FseCState, bits: &mut BitCStream) {
        bits.add_bits(state.value, self.table_log);
        bits.flush();
    }

    /// `FSE_bitCost` at accuracy 8 (256ths of a bit) for `symbol`; `None`
    /// when the symbol has zero probability in this table. (Kept for the
    /// repeat-table cost comparison, not wired up yet.)
    #[allow(dead_code)]
    pub fn bit_cost(&self, symbol: usize) -> Option<u32> {
        if symbol >= self.symbol_tt.len() {
            return None;
        }
        let tt = self.symbol_tt[symbol];
        let table_log = self.table_log;
        let min_nb_bits = tt.delta_nb_bits >> 16;
        let threshold = (min_nb_bits + 1) << 16;
        let bad_cost = (table_log + 1) << 8;
        // Zero-probability symbols were given deltaNbBits = ((tableLog+1)<<16) - tableSize.
        if min_nb_bits > table_log {
            return None;
        }
        let table_size = 1u32 << table_log;
        let delta_from_threshold = threshold.wrapping_sub(tt.delta_nb_bits + table_size);
        let normalized = (delta_from_threshold << 8) >> table_log;
        let bit_multiplier = 1u32 << 8;
        let cost = (min_nb_bits + 1) * bit_multiplier - normalized;
        if cost >= bad_cost {
            return None;
        }
        Some(cost)
    }
}

// ---------------------------------------------------------------------------
// FSE normalization + table log (FSE_normalizeCount / FSE_optimalTableLog)
// ---------------------------------------------------------------------------

pub const FSE_MIN_TABLELOG: u32 = 5;
pub const FSE_MAX_TABLELOG: u32 = 12;

fn fse_min_table_log(src_size: usize, max_symbol_value: u32) -> u32 {
    let min_bits_src = super::seqstore::highbit32(src_size as u32) + 1;
    let min_bits_symbols = super::seqstore::highbit32(max_symbol_value) + 2;
    min_bits_src.min(min_bits_symbols)
}

/// `FSE_optimalTableLog_internal`.
pub fn fse_optimal_table_log(max_table_log: u32, src_size: usize, max_symbol_value: u32, minus: u32) -> u32 {
    debug_assert!(src_size > 1);
    let max_bits_src = super::seqstore::highbit32((src_size - 1) as u32).saturating_sub(minus);
    let mut table_log = max_table_log;
    let min_bits = fse_min_table_log(src_size, max_symbol_value);
    if max_bits_src < table_log {
        table_log = max_bits_src;
    }
    if min_bits > table_log {
        table_log = min_bits;
    }
    table_log.clamp(FSE_MIN_TABLELOG, FSE_MAX_TABLELOG)
}

/// `FSE_normalizeCount`. `count` has `max_symbol_value + 1` entries whose
/// sum is `total`. Returns `None` for the RLE case or on failure.
pub fn fse_normalize_count(count: &[u32], total: usize, table_log: u32, use_low_prob_count: bool) -> Option<Vec<i16>> {
    let max_symbol_value = count.len() - 1;
    if table_log < fse_min_table_log(total, max_symbol_value as u32) {
        return None;
    }
    const RTB_TABLE: [u32; 8] = [0, 473195, 504333, 520860, 550000, 700000, 750000, 830000];
    let low_prob_count: i16 = if use_low_prob_count { -1 } else { 1 };
    let scale = 62 - table_log;
    let step: u64 = (1u64 << 62) / total as u64;
    let v_step: u64 = 1u64 << (scale - 20);
    let mut still_to_distribute: i32 = 1 << table_log;
    let mut largest = 0usize;
    let mut largest_p: i16 = 0;
    let low_threshold = (total >> table_log) as u32;
    let mut norm = vec![0i16; count.len()];

    for s in 0..=max_symbol_value {
        let c = count[s];
        if c as usize == total {
            return None; // RLE special case
        }
        if c == 0 {
            norm[s] = 0;
            continue;
        }
        if c <= low_threshold {
            norm[s] = low_prob_count;
            still_to_distribute -= 1;
        } else {
            let mut proba = ((c as u64 * step) >> scale) as i16;
            if proba < 8 {
                let rest_to_beat = v_step * RTB_TABLE[proba as usize] as u64;
                proba += (((c as u64 * step) - ((proba as u64) << scale)) > rest_to_beat) as i16;
            }
            if proba > largest_p {
                largest_p = proba;
                largest = s;
            }
            norm[s] = proba;
            still_to_distribute -= proba as i32;
        }
    }
    if -still_to_distribute >= (norm[largest] as i32 >> 1) {
        normalize_m2(&mut norm, table_log, count, total, low_prob_count)?;
    } else {
        norm[largest] += still_to_distribute as i16;
    }
    Some(norm)
}

fn normalize_m2(norm: &mut [i16], table_log: u32, count: &[u32], mut total: usize, low_prob_count: i16) -> Option<()> {
    const NOT_YET_ASSIGNED: i16 = -2;
    let max_symbol_value = count.len() - 1;
    let mut distributed: u32 = 0;
    let low_threshold = (total >> table_log) as u32;
    let mut low_one = ((total * 3) >> (table_log + 1)) as u32;

    for s in 0..=max_symbol_value {
        let c = count[s];
        if c == 0 {
            norm[s] = 0;
            continue;
        }
        if c <= low_threshold {
            norm[s] = low_prob_count;
            distributed += 1;
            total -= c as usize;
            continue;
        }
        if c <= low_one {
            norm[s] = 1;
            distributed += 1;
            total -= c as usize;
            continue;
        }
        norm[s] = NOT_YET_ASSIGNED;
    }
    let mut to_distribute = (1u32 << table_log) - distributed;
    if to_distribute == 0 {
        return Some(());
    }
    if (total as u32 / to_distribute) > low_one {
        low_one = ((total * 3) / (to_distribute as usize * 2)) as u32;
        for s in 0..=max_symbol_value {
            if norm[s] == NOT_YET_ASSIGNED && count[s] <= low_one {
                norm[s] = 1;
                distributed += 1;
                total -= count[s] as usize;
            }
        }
        to_distribute = (1u32 << table_log) - distributed;
    }
    if distributed as usize == max_symbol_value + 1 {
        let mut max_v = 0usize;
        let mut max_c = 0u32;
        for s in 0..=max_symbol_value {
            if count[s] > max_c {
                max_v = s;
                max_c = count[s];
            }
        }
        norm[max_v] += to_distribute as i16;
        return Some(());
    }
    if total == 0 {
        let mut s = 0usize;
        while to_distribute > 0 {
            if norm[s] > 0 {
                to_distribute -= 1;
                norm[s] += 1;
            }
            s = (s + 1) % (max_symbol_value + 1);
        }
        return Some(());
    }
    let v_step_log = 62 - table_log as u64;
    let mid = (1u64 << (v_step_log - 1)) - 1;
    let r_step = (((1u64 << v_step_log) * to_distribute as u64) + mid) / total as u64;
    let mut tmp_total = mid;
    for s in 0..=max_symbol_value {
        if norm[s] == NOT_YET_ASSIGNED {
            let end = tmp_total + count[s] as u64 * r_step;
            let s_start = (tmp_total >> v_step_log) as u32;
            let s_end = (end >> v_step_log) as u32;
            let weight = s_end - s_start;
            if weight < 1 {
                return None;
            }
            norm[s] = weight as i16;
            tmp_total = end;
        }
    }
    Some(())
}

/// `FSE_writeNCount`: serialize a normalized distribution. Returns the bytes
/// appended to `out`.
pub fn fse_write_ncount(out: &mut Vec<u8>, norm: &[i16], table_log: u32) -> usize {
    let max_symbol_value = norm.len() - 1;
    let table_size = 1u32 << table_log;
    let mut bits = BitCStream::new(out, 2 * norm.len() + 8);
    bits.add_bits(table_log - 5, 4);
    let mut remaining: i32 = table_size as i32 + 1;
    let mut threshold: i32 = table_size as i32;
    let mut nb_bits: u32 = table_log + 1;
    let mut symbol = 0usize;
    let mut previous_is_0 = false;
    while remaining > 1 && symbol <= max_symbol_value {
        if previous_is_0 {
            let mut start = symbol;
            while symbol <= max_symbol_value && norm[symbol] == 0 {
                symbol += 1;
            }
            if symbol == max_symbol_value + 1 {
                break;
            }
            while symbol >= start + 24 {
                start += 24;
                bits.add_bits(0xFFFF, 16);
                if bits_full(&bits) {
                    bits.flush();
                }
            }
            while symbol >= start + 3 {
                start += 3;
                bits.add_bits(3, 2);
            }
            bits.add_bits((symbol - start) as u32, 2);
            bits.flush();
        }
        let count = norm[symbol] as i32;
        symbol += 1;
        let max = (2 * threshold - 1) - remaining;
        remaining -= count.abs();
        let mut count = count + 1;
        if count >= threshold {
            count += max;
        }
        bits.add_bits(count as u32, if count < max { nb_bits - 1 } else { nb_bits });
        previous_is_0 = count == 1;
        if remaining < 1 {
            break;
        }
        while remaining < threshold {
            nb_bits -= 1;
            threshold >>= 1;
        }
        bits.flush();
    }
    bits.close_no_sentinel()
}

#[inline(always)]
fn bits_full(b: &BitCStream) -> bool {
    b.bit_pos > 48
}
