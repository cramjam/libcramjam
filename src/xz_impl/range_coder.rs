//! LZMA range coder.
//!
//! Direct port of liblzma's `range_decoder.h` and `range_encoder.h`.  The
//! constants come from `range_common.h`:
//!
//! ```text
//! RC_SHIFT_BITS         = 8
//! RC_TOP_BITS           = 24
//! RC_TOP_VALUE          = 1 << RC_TOP_BITS
//! RC_BIT_MODEL_TOTAL_BITS = 11
//! RC_BIT_MODEL_TOTAL    = 1 << RC_BIT_MODEL_TOTAL_BITS
//! RC_MOVE_BITS          = 5
//! ```
//!
//! Probabilities are 16-bit unsigned values in `[0, RC_BIT_MODEL_TOTAL]`.
//! "Reset" sets every probability to `RC_BIT_MODEL_TOTAL >> 1` (i.e., 0.5).

use std::io;

pub const RC_SHIFT_BITS: u32 = 8;
pub const RC_TOP_BITS: u32 = 24;
pub const RC_TOP_VALUE: u32 = 1 << RC_TOP_BITS;
pub const RC_BIT_MODEL_TOTAL_BITS: u32 = 11;
pub const RC_BIT_MODEL_TOTAL: u32 = 1 << RC_BIT_MODEL_TOTAL_BITS;
pub const RC_MOVE_BITS: u32 = 5;

/// Probability type, 16-bit per liblzma.
pub type Prob = u16;

#[inline]
pub fn prob_init() -> Prob {
    (RC_BIT_MODEL_TOTAL >> 1) as Prob
}

/// Reset every probability in a slice to 0.5.
pub fn prob_reset_slice(probs: &mut [Prob]) {
    for p in probs.iter_mut() {
        *p = prob_init();
    }
}

// =========================================================================
// Range Decoder
// =========================================================================

/// LZMA range decoder.  Reads from an arbitrary `&[u8]` slice via `pos`.
pub struct RangeDecoder<'a> {
    pub range: u32,
    pub code: u32,
    pub input: &'a [u8],
    pub pos: usize,
    /// One past the last valid compressed byte. Defaults to `input.len()`;
    /// LZMA2 sets it to the chunk's `compressed_size` so a corrupt chunk
    /// cannot make the range decoder consume the following chunk's bytes
    /// (or read past the allocation). See `Rc::normalize`.
    pub end: usize,
}

impl<'a> RangeDecoder<'a> {
    /// Initialize a fresh range decoder by consuming the first 5 input
    /// bytes (the LZMA spec mandates the very first byte be 0x00).
    pub fn new(input: &'a [u8]) -> io::Result<Self> {
        if input.len() < 5 {
            return Err(io::Error::new(
                io::ErrorKind::UnexpectedEof,
                "lzma: not enough bytes to initialize range decoder",
            ));
        }
        if input[0] != 0x00 {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "lzma: first range coder byte must be 0x00",
            ));
        }
        let mut code = 0u32;
        for i in 1..5 {
            code = (code << 8) | input[i] as u32;
        }
        Ok(Self {
            range: u32::MAX,
            code,
            input,
            pos: 5,
            end: input.len(),
        })
    }

    /// True iff the range decoder is in the "finished" state expected at
    /// end-of-stream (range coder properly closed; code reduced to zero).
    pub fn is_finished(&self) -> bool {
        self.code == 0
    }

    /// Refill the top bits of `range` if it has shrunk below `RC_TOP_VALUE`.
    #[inline(always)]
    fn normalize(&mut self) -> io::Result<()> {
        if self.range < RC_TOP_VALUE {
            if self.pos >= self.input.len() {
                return Err(io::Error::new(
                    io::ErrorKind::UnexpectedEof,
                    "lzma: range decoder ran out of input",
                ));
            }
            self.range <<= RC_SHIFT_BITS;
            self.code = (self.code << RC_SHIFT_BITS) | self.input[self.pos] as u32;
            self.pos += 1;
        }
        Ok(())
    }

    /// Decode a single binary symbol against probability `prob`, updating
    /// `prob` in place.  Returns the decoded bit (0 or 1).
    #[inline(always)]
    pub fn decode_bit(&mut self, prob: &mut Prob) -> io::Result<u32> {
        self.normalize()?;
        let bound = (self.range >> RC_BIT_MODEL_TOTAL_BITS) * (*prob as u32);
        if self.code < bound {
            self.range = bound;
            *prob = (*prob as u32 + ((RC_BIT_MODEL_TOTAL - *prob as u32) >> RC_MOVE_BITS)) as Prob;
            Ok(0)
        } else {
            self.range -= bound;
            self.code -= bound;
            *prob = (*prob as u32 - (*prob as u32 >> RC_MOVE_BITS)) as Prob;
            Ok(1)
        }
    }

    /// "Fast path" bit decode used inside a hot loop where the caller has
    /// already verified there's enough input padding to refill the range
    /// coder without going past the end.  No `Result`, no `?`-overhead.
    /// `pos < input.len()` is a debug invariant.
    #[inline(always)]
    pub fn decode_bit_fast(&mut self, prob: &mut Prob) -> u32 {
        // Inline normalize() but skip the EOF check.
        if self.range < RC_TOP_VALUE {
            debug_assert!(self.pos < self.input.len(), "decode_bit_fast called with empty input");
            self.range <<= RC_SHIFT_BITS;
            self.code = (self.code << RC_SHIFT_BITS) | self.input[self.pos] as u32;
            self.pos += 1;
        }
        let bound = (self.range >> RC_BIT_MODEL_TOTAL_BITS) * (*prob as u32);
        if self.code < bound {
            self.range = bound;
            *prob = (*prob as u32 + ((RC_BIT_MODEL_TOTAL - *prob as u32) >> RC_MOVE_BITS)) as Prob;
            0
        } else {
            self.range -= bound;
            self.code -= bound;
            *prob = (*prob as u32 - (*prob as u32 >> RC_MOVE_BITS)) as Prob;
            1
        }
    }

    /// Forward bit-tree decode (fast path).  Caller must have padding.
    #[inline(always)]
    pub fn decode_bittree_fast(&mut self, probs: &mut [Prob], num_bits: u32) -> u32 {
        let mut symbol: u32 = 1;
        for _ in 0..num_bits {
            let bit = self.decode_bit_fast(&mut probs[symbol as usize]);
            symbol = (symbol << 1) | bit;
        }
        symbol - (1 << num_bits)
    }

    /// Reverse bit-tree decode (fast path).  Caller must have padding.
    #[inline(always)]
    pub fn decode_bittree_reverse_fast(&mut self, probs: &mut [Prob], num_bits: u32) -> u32 {
        let mut symbol: u32 = 1;
        let mut result: u32 = 0;
        for i in 0..num_bits {
            let bit = self.decode_bit_fast(&mut probs[symbol as usize]);
            symbol = (symbol << 1) | bit;
            result |= bit << i;
        }
        result
    }

    /// Direct (uniform) bits decode (fast path).  Caller must have padding.
    #[inline(always)]
    pub fn decode_direct_bits_fast(&mut self, num_bits: u32) -> u32 {
        let mut result: u32 = 0;
        for _ in 0..num_bits {
            if self.range < RC_TOP_VALUE {
                debug_assert!(self.pos < self.input.len());
                self.range <<= RC_SHIFT_BITS;
                self.code = (self.code << RC_SHIFT_BITS) | self.input[self.pos] as u32;
                self.pos += 1;
            }
            self.range >>= 1;
            let t: u32 = (self.code.wrapping_sub(self.range) as i32 >> 31) as u32;
            self.code = self.code.wrapping_sub(self.range & !t);
            result = (result << 1) | (t.wrapping_add(1) & 1);
        }
        result
    }

    /// Decode `num_bits` from a bit-tree of size `2^num_bits` (forward —
    /// MSB first).  This is the standard LZMA "bittree decode" used for
    /// length sub-coders and the high bits of distance slots.
    pub fn decode_bittree(&mut self, probs: &mut [Prob], num_bits: u32) -> io::Result<u32> {
        debug_assert_eq!(probs.len(), 1usize << num_bits);
        let mut symbol: u32 = 1;
        for _ in 0..num_bits {
            let bit = self.decode_bit(&mut probs[symbol as usize])?;
            symbol = (symbol << 1) | bit;
        }
        Ok(symbol - (1 << num_bits))
    }

    /// Decode `num_bits` from a bit-tree, REVERSE order (LSB first).  Used
    /// for the alignment-bits sub-coder and the low bits of distances.
    pub fn decode_bittree_reverse(
        &mut self,
        probs: &mut [Prob],
        num_bits: u32,
    ) -> io::Result<u32> {
        debug_assert_eq!(probs.len(), 1usize << num_bits);
        let mut symbol: u32 = 1;
        let mut result: u32 = 0;
        for i in 0..num_bits {
            let bit = self.decode_bit(&mut probs[symbol as usize])?;
            symbol = (symbol << 1) | bit;
            result |= bit << i;
        }
        Ok(result)
    }

    /// Decode `num_bits` raw bits with no probability model (uniform).
    /// Used for the middle "direct bits" of large distance slots.
    pub fn decode_direct_bits(&mut self, num_bits: u32) -> io::Result<u32> {
        let mut result: u32 = 0;
        for _ in 0..num_bits {
            self.normalize()?;
            self.range >>= 1;
            let t: u32 = (self.code.wrapping_sub(self.range) as i32 >> 31) as u32;
            // t is now 0xFFFFFFFF if code < range, else 0.
            self.code = self.code.wrapping_sub(self.range & !t);
            result = (result << 1) | (t.wrapping_add(1) & 1);
        }
        Ok(result)
    }
}

// =========================================================================
// Range Encoder
// =========================================================================

/// LZMA range encoder.  Buffers symbols into an output Vec.
pub struct RangeEncoder {
    pub low: u64,
    pub range: u32,
    pub cache_size: u32,
    pub cache: u8,
    pub output: Vec<u8>,
}

impl RangeEncoder {
    pub fn new() -> Self {
        Self::with_capacity(0)
    }

    pub fn with_capacity(cap: usize) -> Self {
        Self {
            low: 0,
            range: u32::MAX,
            cache_size: 1,
            cache: 0,
            output: Vec::with_capacity(cap),
        }
    }

    /// Reset to a fresh empty encoder, reusing the existing output buffer.
    pub fn reset(&mut self) {
        self.low = 0;
        self.range = u32::MAX;
        self.cache_size = 1;
        self.cache = 0;
        self.output.clear();
    }

    /// Push the renormalized top byte to the output as needed.
    fn shift_low(&mut self) {
        // If the top byte of `low` is settled (no carry can change it),
        // emit `cache + 0xFF*cache_size`-style bytes; otherwise propagate
        // the carry through any deferred 0xFF bytes.
        if (self.low as u32) < 0xFF000000 || self.low >> 32 != 0 {
            let mut temp = self.cache;
            loop {
                self.output.push(temp.wrapping_add((self.low >> 32) as u8));
                temp = 0xFF;
                self.cache_size -= 1;
                if self.cache_size == 0 {
                    break;
                }
            }
            self.cache = ((self.low >> 24) & 0xFF) as u8;
            self.cache_size = 1;
        } else {
            self.cache_size += 1;
        }
        self.low = (self.low << 8) & 0xFFFFFFFF;
    }

    /// Refill the top bits of `range` if it has shrunk below `RC_TOP_VALUE`.
    #[inline]
    fn normalize(&mut self) {
        if self.range < RC_TOP_VALUE {
            self.range <<= RC_SHIFT_BITS;
            self.shift_low();
        }
    }

    /// Encode a single binary symbol against probability `prob`, updating
    /// `prob` in place.
    #[inline]
    pub fn encode_bit(&mut self, prob: &mut Prob, bit: u32) {
        let bound = (self.range >> RC_BIT_MODEL_TOTAL_BITS) * (*prob as u32);
        if bit == 0 {
            self.range = bound;
            *prob = (*prob as u32 + ((RC_BIT_MODEL_TOTAL - *prob as u32) >> RC_MOVE_BITS)) as Prob;
        } else {
            self.low += bound as u64;
            self.range -= bound;
            *prob = (*prob as u32 - (*prob as u32 >> RC_MOVE_BITS)) as Prob;
        }
        self.normalize();
    }

    /// Encode `num_bits` of `symbol` MSB-first against a bittree.
    pub fn encode_bittree(&mut self, probs: &mut [Prob], num_bits: u32, symbol: u32) {
        debug_assert_eq!(probs.len(), 1usize << num_bits);
        let mut tree_index: u32 = 1;
        for i in (0..num_bits).rev() {
            let bit = (symbol >> i) & 1;
            self.encode_bit(&mut probs[tree_index as usize], bit);
            tree_index = (tree_index << 1) | bit;
        }
    }

    /// Encode `num_bits` of `symbol` LSB-first against a bittree.
    pub fn encode_bittree_reverse(&mut self, probs: &mut [Prob], num_bits: u32, symbol: u32) {
        debug_assert_eq!(probs.len(), 1usize << num_bits);
        let mut tree_index: u32 = 1;
        let mut sym = symbol;
        for _ in 0..num_bits {
            let bit = sym & 1;
            sym >>= 1;
            self.encode_bit(&mut probs[tree_index as usize], bit);
            tree_index = (tree_index << 1) | bit;
        }
    }

    /// Encode `num_bits` raw bits (uniform), MSB first.
    pub fn encode_direct_bits(&mut self, value: u32, num_bits: u32) {
        for i in (0..num_bits).rev() {
            self.range >>= 1;
            if (value >> i) & 1 != 0 {
                self.low += self.range as u64;
            }
            self.normalize();
        }
    }

    /// Flush the remaining range coder bits.  After this, `output` is the
    /// final byte stream.
    pub fn finish(&mut self) {
        for _ in 0..5 {
            self.shift_low();
        }
    }
}

impl Default for RangeEncoder {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Round-trip a fixed sequence of bits through encode + decode.
    /// Note: the encoder is responsible for producing the leading 0x00 byte
    /// the decoder requires; we pass its output through verbatim.
    #[test]
    fn round_trip_bits() {
        let mut enc_probs = vec![prob_init(); 256];
        let mut dec_probs = vec![prob_init(); 256];

        let mut rc = RangeEncoder::new();
        let bits: Vec<(usize, u32)> = (0..500)
            .map(|i| (i % 256, (((i * 7) ^ (i * 13 >> 1)) & 1) as u32))
            .collect();

        for &(p, b) in &bits {
            rc.encode_bit(&mut enc_probs[p], b);
        }
        rc.finish();
        let encoded = rc.output.clone();
        assert!(!encoded.is_empty(), "encoder produced no output");
        assert_eq!(encoded[0], 0x00, "encoder must produce a leading 0x00 byte");

        let mut rd = RangeDecoder::new(&encoded).unwrap();
        for &(p, expected) in &bits {
            let actual = rd.decode_bit(&mut dec_probs[p]).unwrap();
            assert_eq!(actual, expected, "bit at prob {} mismatched", p);
        }
    }

    #[test]
    fn round_trip_bittrees_and_direct() {
        let mut enc_tree = vec![prob_init(); 16];
        let mut dec_tree = vec![prob_init(); 16];
        let mut enc_rev = vec![prob_init(); 8];
        let mut dec_rev = vec![prob_init(); 8];

        let symbols: &[u32] = &[0, 7, 15, 3, 12, 1, 10];
        let rev_syms: &[u32] = &[5, 2, 6, 0, 7, 3];
        let direct: &[u32] = &[0xABCD, 0x1234, 0xFFFF, 0x0000];

        let mut rc = RangeEncoder::new();
        for &s in symbols {
            rc.encode_bittree(&mut enc_tree, 4, s);
        }
        for &s in rev_syms {
            rc.encode_bittree_reverse(&mut enc_rev, 3, s);
        }
        for &v in direct {
            rc.encode_direct_bits(v, 16);
        }
        rc.finish();

        let mut rd = RangeDecoder::new(&rc.output).unwrap();
        for &expected in symbols {
            assert_eq!(rd.decode_bittree(&mut dec_tree, 4).unwrap(), expected);
        }
        for &expected in rev_syms {
            assert_eq!(rd.decode_bittree_reverse(&mut dec_rev, 3).unwrap(), expected);
        }
        for &expected in direct {
            assert_eq!(rd.decode_direct_bits(16).unwrap(), expected);
        }
    }
}
