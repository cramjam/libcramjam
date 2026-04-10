//! FSE (Finite State Entropy) decoding for Zstandard (RFC 8878 Section 4.1).
//!
//! FSE is a table-based ANS (Asymmetric Numeral Systems) codec. Each entry in
//! the decoding table contains: symbol, number of bits to read, and a baseline
//! for computing the next state.

use std::io;

use super::bits::{ForwardBitWriter, ForwardByteReader, ReverseBitReader};

/// A single entry in an FSE decoding table.
#[derive(Clone, Copy, Default)]
pub struct FseEntry {
    pub symbol: u8,
    pub num_bits: u8,
    pub baseline: u16,
}

/// FSE decoding table.
pub struct FseTable {
    pub table: Vec<FseEntry>,
    pub accuracy_log: u32,
}

impl FseTable {
    /// Build a decoding table from a distribution of symbol weights.
    ///
    /// `norm_weights[symbol]` is the normalized probability for that symbol.
    /// A weight of -1 means "less than 1" (low-probability symbol).
    pub fn from_weights(weights: &[i16], accuracy_log: u32) -> io::Result<Self> {
        let table_size = 1usize << accuracy_log;
        let mut table = vec![FseEntry::default(); table_size];

        // Phase 1: Spread symbols across the table.
        // Symbols with weight == -1 get exactly 1 slot, placed at the end.
        // Other symbols get `weight` slots, distributed evenly.
        let mut high_threshold = table_size;
        for (sym, &w) in weights.iter().enumerate() {
            if w == -1 {
                high_threshold -= 1;
                table[high_threshold].symbol = sym as u8;
            }
        }

        let step = (table_size >> 1) + (table_size >> 3) + 3;
        let mask = table_size - 1;
        let mut pos = 0usize;

        for (sym, &w) in weights.iter().enumerate() {
            if w <= 0 {
                continue;
            }
            for _ in 0..w as usize {
                table[pos].symbol = sym as u8;
                // Advance position, skipping slots already used by high_threshold symbols.
                loop {
                    pos = (pos + step) & mask;
                    if pos < high_threshold {
                        break;
                    }
                }
            }
        }

        // Phase 2: Compute num_bits and baseline for each entry.
        let mut symbol_next = vec![0u16; weights.len()];
        for (sym, &w) in weights.iter().enumerate() {
            let s = if w == -1 { 1 } else { w as u16 };
            symbol_next[sym] = s;
        }

        for i in 0..table_size {
            let sym = table[i].symbol as usize;
            let s = symbol_next[sym];
            let nb = (accuracy_log - highest_bit(s as u32)) as u8;
            table[i].num_bits = nb;
            table[i].baseline = ((s as u32) << nb) as u16 - table_size as u16;
            symbol_next[sym] = s + 1;
        }

        Ok(Self {
            table,
            accuracy_log,
        })
    }

    /// Decode the FSE table description from a compressed bitstream.
    ///
    /// Returns the table and the number of bytes consumed.
    /// Decode FSE table description following the exact algorithm from
    /// the zstd reference (FSE_readNCount).
    pub fn decode_table(reader: &mut ForwardByteReader, max_symbol: u32, max_accuracy_log: u32) -> io::Result<Self> {
        let data = &reader.data[reader.pos..];
        if data.len() < 4 {
            // Need at least 4 bytes for the LE u32 load.
            if data.is_empty() {
                return Err(io::Error::new(io::ErrorKind::InvalidData, "zstd: empty FSE table"));
            }
        }

        // Load the first 4 bytes as LE u32 (zero-padded if < 4 bytes).
        let load_le32 = |data: &[u8], byte_pos: usize| -> u32 {
            let mut buf = [0u8; 4];
            let end = data.len().min(byte_pos + 4);
            let start = byte_pos.min(end);
            buf[..end - start].copy_from_slice(&data[start..end]);
            u32::from_le_bytes(buf)
        };

        let mut bit_stream = load_le32(data, 0);
        let accuracy_log = (bit_stream & 0xF) + 5;
        if accuracy_log > max_accuracy_log {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "zstd: FSE accuracy log exceeds maximum",
            ));
        }
        bit_stream >>= 4;
        let mut bit_count = 4u32;

        let table_size = 1u32 << accuracy_log;
        let mut remaining = table_size as i32 + 1;
        let mut threshold = table_size as i32;
        let mut nb_bits = accuracy_log + 1;
        #[cfg(test)]
        eprintln!("[fse_init] accuracy_log={accuracy_log} table_size={table_size} remaining={remaining} threshold={threshold} nb_bits={nb_bits}");
        let mut weights: Vec<i16> = Vec::new();
        let max_sym = max_symbol as usize + 1;

        while remaining > 1 && weights.len() < max_sym {
            // Adjust threshold/nbBits for current remaining.
            while remaining < threshold && nb_bits > 1 {
                nb_bits -= 1;
                threshold >>= 1;
            }

            // Maybe reload bitStream.
            if bit_count >= 16 {
                let byte_pos = (bit_count >> 3) as usize;
                bit_stream = load_le32(data, byte_pos);
                bit_stream >>= bit_count & 7;
            }

            let max_val = (2 * threshold - 1) - remaining;
            let low = bit_stream & (threshold - 1) as u32;

            let count;
            if (low as i32) < max_val {
                // Short code: only nb_bits-1 bits consumed.
                count = low as i32;
                bit_count += nb_bits - 1;
                bit_stream >>= nb_bits - 1;
            } else {
                // Long code: nb_bits bits consumed.
                let full = bit_stream & (2 * threshold as u32 - 1);
                count = if full as i32 >= threshold {
                    full as i32 - max_val
                } else {
                    full as i32
                };
                bit_count += nb_bits;
                bit_stream >>= nb_bits;
            }

            let prob = count - 1; // -1 means "less than 1"
            weights.push(prob as i16);

            #[cfg(test)]
            eprintln!("[fse] sym={} remaining={remaining} threshold={threshold} nb_bits={nb_bits} max_val={max_val} low={low} count={count} prob={prob}",
                weights.len()-1);

            remaining -= if prob < 0 { 1 } else { prob as i32 };

            // Handle repeat-zero encoding.
            if prob == 0 {
                loop {
                    if bit_count >= 16 {
                        let byte_pos = (bit_count >> 3) as usize;
                        bit_stream = load_le32(data, byte_pos);
                        bit_stream >>= bit_count & 7;
                    }
                    let repeat = (bit_stream & 3) as usize;
                    bit_stream >>= 2;
                    bit_count += 2;
                    for _ in 0..repeat {
                        weights.push(0);
                    }
                    if repeat < 3 {
                        break;
                    }
                }
            }
        }

        // Fill remaining symbols with 0.
        while weights.len() < max_sym {
            weights.push(0);
        }

        let bytes_consumed = ((bit_count + 7) >> 3) as usize;
        reader.pos += bytes_consumed;

        Self::from_weights(&weights, accuracy_log)
    }

    /// Peek at the symbol for the current state.
    #[inline(always)]
    pub fn symbol(&self, state: u32) -> u8 {
        self.table[state as usize].symbol
    }

    /// Advance to the next state by reading bits from the backward bitstream.
    #[inline(always)]
    pub fn next_state(&self, state: u32, bits: &mut ReverseBitReader) -> u32 {
        let entry = &self.table[state as usize];
        let low_bits = bits.get_bits(entry.num_bits as u32);
        entry.baseline as u32 + low_bits
    }
}

/// Build an FSE table from a single repeated symbol (RLE mode).
pub fn fse_rle_table(symbol: u8) -> FseTable {
    FseTable {
        table: vec![FseEntry {
            symbol,
            num_bits: 0,
            baseline: 0,
        }],
        accuracy_log: 0,
    }
}

/// Highest set bit position (0-indexed). `highest_bit(1) = 0, highest_bit(4) = 2`.
#[inline(always)]
fn highest_bit(v: u32) -> u32 {
    debug_assert!(v > 0);
    31 - v.leading_zeros()
}

// ---------------------------------------------------------------------------
// Predefined FSE tables (RFC 8878 Appendix A)
// ---------------------------------------------------------------------------

/// Predefined Literals Length weights (RFC 8878 Appendix A, Table 14).
pub(crate) static PREDEFINED_LL_WEIGHTS: [i16; 36] = [
    4, 3, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 1, 1, 1,
    2, 2, 2, 2, 2, 2, 2, 2, 2, 3, 2, 1, 1, 1, 1, 1,
    -1, -1, -1, -1,
];

/// Predefined Match Length weights (RFC 8878 Appendix A, Table 16).
pub(crate) static PREDEFINED_ML_WEIGHTS: [i16; 53] = [
    1, 4, 3, 2, 2, 2, 2, 2, 2, 1, 1, 1, 1, 1, 1, 1,
    1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
    1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, -1, -1,
    -1, -1, -1, -1, -1,
];

/// Predefined Offset weights (RFC 8878 Appendix A, Table 18).
pub(crate) static PREDEFINED_OF_WEIGHTS: [i16; 29] = [
    1, 1, 1, 1, 1, 1, 2, 2, 2, 1, 1, 1, 1, 1, 1, 1,
    1, 1, 1, 1, 1, 1, 1, 1, -1, -1, -1, -1, -1,
];

/// Build the predefined Literals Length FSE table (accuracy_log = 6).
pub fn predefined_litlen_table() -> FseTable {
    FseTable::from_weights(&PREDEFINED_LL_WEIGHTS, 6).unwrap()
}

/// Build the predefined Match Length FSE table (accuracy_log = 6).
pub fn predefined_matchlen_table() -> FseTable {
    FseTable::from_weights(&PREDEFINED_ML_WEIGHTS, 6).unwrap()
}

/// Build the predefined Offset FSE table (accuracy_log = 5).
pub fn predefined_offset_table() -> FseTable {
    FseTable::from_weights(&PREDEFINED_OF_WEIGHTS, 5).unwrap()
}

// ---------------------------------------------------------------------------
// Value decoding tables for literals length, match length, and offset
// (RFC 8878 Section 3.1.2.1.2)
// ---------------------------------------------------------------------------

/// (baseline, extra_bits) for Literals_Length codes 0..35.
pub const LITLEN_TABLE: [(u32, u8); 36] = [
    (0, 0), (1, 0), (2, 0), (3, 0), (4, 0), (5, 0), (6, 0), (7, 0),
    (8, 0), (9, 0), (10, 0), (11, 0), (12, 0), (13, 0), (14, 0), (15, 0),
    (16, 1), (18, 1), (20, 1), (22, 1), (24, 2), (28, 2), (32, 3), (40, 3),
    (48, 4), (64, 6), (128, 7), (256, 8), (512, 9), (1024, 10), (2048, 11), (4096, 12),
    (8192, 13), (16384, 14), (32768, 15), (65536, 16),
];

/// (baseline, extra_bits) for Match_Length codes 0..52.
pub const MATCHLEN_TABLE: [(u32, u8); 53] = [
    (3, 0), (4, 0), (5, 0), (6, 0), (7, 0), (8, 0), (9, 0), (10, 0),
    (11, 0), (12, 0), (13, 0), (14, 0), (15, 0), (16, 0), (17, 0), (18, 0),
    (19, 0), (20, 0), (21, 0), (22, 0), (23, 0), (24, 0), (25, 0), (26, 0),
    (27, 0), (28, 0), (29, 0), (30, 0), (31, 0), (32, 0), (33, 0), (34, 0),
    (35, 1), (37, 1), (39, 1), (41, 1), (43, 2), (47, 2), (51, 3), (59, 3),
    (67, 4), (83, 4), (99, 5), (131, 7), (259, 8), (515, 9), (1027, 10), (2051, 11),
    (4099, 12), (8195, 13), (16387, 14), (32771, 15), (65539, 16),
];

// ---------------------------------------------------------------------------
// FSE encoder side
// ---------------------------------------------------------------------------

/// One encoder-side state slot for a particular symbol.  These are derived
/// from the decoder table by inverting the (state → symbol) mapping.
#[derive(Clone, Copy, Debug)]
struct FseEncoderState {
    /// The decoder-table index this slot transitions INTO.  Becomes the new
    /// `state` value after encoding the symbol.
    index: u16,
    /// `num_bits` worth of low bits of the *previous* state are written to the
    /// stream when transitioning here.  Equal to the decoder entry's num_bits.
    num_bits: u8,
    /// Inclusive range of *previous* state indexes that route to this slot.
    /// `prev_state ∈ [base..=last]` chooses this slot when encoding the symbol.
    base: u32,
    last: u32,
}

/// Encoder-side FSE table: per-symbol list of state slots.
pub struct FseEncoder {
    /// `slots[symbol]` holds the encoder slots for that symbol, sorted by `index`.
    slots: Vec<Vec<FseEncoderState>>,
    pub accuracy_log: u32,
}

impl FseEncoder {
    /// Build an encoder table from a decoder table.
    pub fn from_decoder(dec: &FseTable, num_symbols: usize) -> Self {
        let acc_log = dec.accuracy_log;
        let table_size = 1usize << acc_log;
        let mut slots: Vec<Vec<FseEncoderState>> = vec![Vec::new(); num_symbols];

        // Each decoder entry says: at decoder state `i`, the symbol is `sym`
        // and the next state is `baseline + read_bits(num_bits)`.  Inverting:
        // when we ENCODE `sym`, we need to land on entry `i`.  The set of
        // *prior* states that select entry `i` is exactly
        //   prev_state ∈ [baseline + 0 .. baseline + (1<<num_bits) - 1]
        // which is what the decoder will output.
        for i in 0..table_size {
            let entry = dec.table[i];
            let sym = entry.symbol as usize;
            let nb = entry.num_bits as u32;
            let base = entry.baseline as u32;
            let span = 1u32 << nb;
            slots[sym].push(FseEncoderState {
                index: i as u16,
                num_bits: nb as u8,
                base,
                last: base + span - 1,
            });
        }

        // Sort each symbol's slot list by `base` so the lookup is O(log n)
        // (or even O(1) given small list sizes).
        for s in slots.iter_mut() {
            s.sort_by_key(|st| st.base);
        }

        Self {
            slots,
            accuracy_log: acc_log,
        }
    }

    /// Initial encoder state for the FIRST symbol to be encoded (which is
    /// the LAST symbol in input order).  By zstd convention, use the slot
    /// with the smallest `index` for that symbol.
    pub fn start_state(&self, sym: u8) -> u32 {
        self.slots[sym as usize][0].index as u32
    }

    /// Encode one symbol: write the low bits of `prev_state`, return the new
    /// state.  Caller must have at least `symbol_count(sym) > 0` for this to
    /// be a valid encoding (i.e. the FSE table must contain `sym`).
    pub fn encode_symbol(&self, prev_state: u32, sym: u8, w: &mut ForwardBitWriter) -> u32 {
        let slot = self.find_slot(sym, prev_state);
        let diff = prev_state - slot.base;
        w.write_bits(diff as u64, slot.num_bits as u32);
        slot.index as u32
    }

    /// Find the slot whose [base..=last] contains `prev_state`.
    /// Slots are sorted by `base`, but the predefined-table slot ranges
    /// can WRAP around table_size, so the simple linear scan must check
    /// each candidate explicitly.
    fn find_slot(&self, sym: u8, prev_state: u32) -> &FseEncoderState {
        let table_size = 1u32 << self.accuracy_log;
        let slots = &self.slots[sym as usize];
        for s in slots {
            // Range [base..=last] in modular table_size space.
            if s.base <= s.last {
                if prev_state >= s.base && prev_state <= s.last {
                    return s;
                }
            } else {
                // Wrapped range (base > last) covers [base..table_size) ∪ [0..=last]
                if prev_state >= s.base && prev_state < table_size {
                    return s;
                }
                if prev_state <= s.last {
                    return s;
                }
            }
        }
        // Should never happen with a well-formed table.
        panic!(
            "FSE encoder: no slot for sym={} prev_state={} (table_size={})",
            sym, prev_state, table_size
        );
    }
}

#[cfg(test)]
mod encoder_tests {
    use super::*;
    use crate::zstd_impl::bits::{ForwardBitWriter, ReverseBitReader};

    /// End-to-end check: predefined LL table encode→decode round trip.
    #[test]
    fn fse_encode_decode_roundtrip_litlen() {
        let dec = predefined_litlen_table();
        let enc = FseEncoder::from_decoder(&dec, 36);

        // Symbols to encode (must all be < 36 and have probability > 0 in the table).
        let input: Vec<u8> = vec![0, 1, 2, 3, 4, 5, 10, 15, 20, 25, 30, 31, 0, 5, 10];

        let mut w = ForwardBitWriter::new();
        // Encode in REVERSE order (decoder reads MSB-first from end → forward).
        let mut state = enc.start_state(*input.last().unwrap());
        for &sym in input.iter().rev().skip(1) {
            state = enc.encode_symbol(state, sym, &mut w);
        }
        // Final state goes last so the decoder reads it first.
        w.write_bits(state as u64, enc.accuracy_log);
        let bytes = w.finalize();

        // Now decode forward and check we recover the input.
        let mut br = ReverseBitReader::new(&bytes).unwrap();
        br.skip_padding_bits().unwrap();
        let mut decoded = Vec::new();
        let mut state = br.get_bits(dec.accuracy_log);
        for _ in 0..input.len() {
            decoded.push(dec.symbol(state));
            // For the LAST symbol decoded we mustn't update state (no more bits).
            if decoded.len() < input.len() {
                state = dec.next_state(state, &mut br);
            }
        }
        assert_eq!(decoded, input, "FSE encode/decode roundtrip mismatch");
    }
}
