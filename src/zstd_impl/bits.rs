//! Bit-level readers for Zstandard.
//!
//! Zstd uses **backward** bitstreams: data is read from the last byte toward
//! the first, MSB-first within each loaded word.  This is the opposite of
//! DEFLATE's LSB-first forward streams.
//!
//! Modeled after ruzstd's BitReaderReversed for proven correctness.

use std::io;

/// Backward bit reader for FSE and Huffman decoding.
///
/// The container is loaded in 8-byte windows from the END of the source backward.
/// `bits_consumed` counts from the **MSB** side: when bits_consumed=0, the next
/// bit returned is the highest bit in the container; once bits_consumed reaches 64,
/// a refill must occur.
///
/// Reading past the start of the source is allowed (returns 0 bits) but increments
/// `extra_bits`, which makes `bits_remaining()` go negative — callers use this as
/// the termination signal for FSE state-update loops.
#[derive(Clone, Copy)]
pub struct ReverseBitReader<'a> {
    /// Index into `source`: bytes [index..index+8] are currently in the container.
    index: usize,
    /// Bits consumed from the high side of `bit_container` (0..=64).
    bits_consumed: u8,
    /// How many bits past the start of input have been "consumed" via padding.
    extra_bits: usize,
    /// Source slice.
    source: &'a [u8],
    /// 64-bit accumulator. Newly loaded bytes occupy the high half (LE-loaded);
    /// after consume/refill the meaningful bits are at the top.
    bit_container: u64,
    /// `source.len() - 7` (or 0 when the source is shorter than 8 bytes):
    /// a refill may take the one-load fast path iff the new window index is
    /// strictly below this. Precomputed so the hot refill is one compare.
    refill_limit: usize,
}

impl<'a> ReverseBitReader<'a> {
    /// Create a new reader.  Container is empty until the first `get_bits` call.
    pub fn new(source: &'a [u8]) -> io::Result<Self> {
        if source.is_empty() {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "zstd: empty bitstream",
            ));
        }
        Ok(Self {
            index: source.len(),
            bits_consumed: 64,
            extra_bits: 0,
            source,
            bit_container: 0,
            refill_limit: if source.len() >= 8 { source.len() - 7 } else { 0 },
        })
    }

    /// Bits remaining (signed).  Negative once the reader has overrun the input.
    #[inline]
    pub fn bits_remaining(&self) -> isize {
        self.index as isize * 8 + (64 - self.bits_consumed as isize) - self.extra_bits as isize
    }

    /// Refill the container so at least 56 bits are available (when possible).
    ///
    /// Split into a hot inlineable fast path and a cold out-of-line tail. The
    /// fast path is a single unaligned 8-byte load — no bounds check beyond
    /// the `>= 8 bytes remaining` guard — and gets inlined into every
    /// `get_bits` call. The cold tail (near start of input, partial loads,
    /// overrun) stays out-of-line so the caller's icache stays clean.
    #[inline(always)]
    pub fn refill(&mut self) {
        let bytes_consumed = (self.bits_consumed / 8) as usize;
        // Fast path: the new window still lies fully inside the source.
        // `wrapping_sub` turns an underflow into a huge value that fails the
        // single compare, so this is one predictable branch.
        let new_index = self.index.wrapping_sub(bytes_consumed);
        if new_index < self.refill_limit {
            self.index = new_index;
            self.bits_consumed &= 7;
            // SAFETY: `new_index + 8 <= source.len()` just checked.
            unsafe {
                let ptr = self.source.as_ptr().add(new_index) as *const u64;
                self.bit_container = u64::from_le(std::ptr::read_unaligned(ptr));
            }
            return;
        }
        *self = self.refill_slow();
    }

    /// Out-of-line slow path for `refill`. Handles the start-of-input region,
    /// partial loads shorter than 8 bytes, and overrun past the stream start.
    /// Takes and returns `self` by value so the hot loop's reader never has
    /// its address escape (which would pin it to the stack).
    #[cold]
    #[inline(never)]
    fn refill_slow(self) -> Self {
        self.refill_slow_body()
    }

    #[inline(always)]
    fn refill_slow_body(mut self) -> Self {
        let bytes_consumed = (self.bits_consumed / 8) as usize;
        if bytes_consumed == 0 {
            return self;
        }

        if self.index >= bytes_consumed {
            self.index -= bytes_consumed;
            self.bits_consumed &= 7;
            // Near the end of source (small input): partial load.
            let mut buf = [0u8; 8];
            let avail = self.source.len() - self.index;
            buf[..avail].copy_from_slice(&self.source[self.index..]);
            self.bit_container = u64::from_le_bytes(buf);
        } else if self.index > 0 {
            // Last partial load: read from offset 0.
            if self.source.len() >= 8 {
                unsafe {
                    let ptr = self.source.as_ptr() as *const u64;
                    self.bit_container = u64::from_le(std::ptr::read_unaligned(ptr));
                }
            } else {
                let mut buf = [0u8; 8];
                buf[..self.source.len()].copy_from_slice(self.source);
                self.bit_container = u64::from_le_bytes(buf);
            }
            self.bits_consumed -= 8 * self.index as u8;
            self.index = 0;
            self.bit_container <<= self.bits_consumed;
            self.extra_bits += self.bits_consumed as usize;
            self.bits_consumed = 0;
            // The container now holds phantom zero bits past the stream
            // start; a fast-path reload at index 0 would discard that
            // shift, so route every later refill through here.
            self.refill_limit = 0;
        } else if self.bits_consumed < 64 {
            self.bit_container <<= self.bits_consumed;
            self.extra_bits += self.bits_consumed as usize;
            self.bits_consumed = 0;
            self.refill_limit = 0;
        } else {
            self.extra_bits += self.bits_consumed as usize;
            self.bits_consumed = 0;
            self.bit_container = 0;
            self.refill_limit = 0;
        }
        self
    }

    /// Read up to 56 bits.
    ///
    /// The `n == 0` case is handled via `(1u64 << 0) - 1 = 0` masking: we
    /// never early-return, since in release builds `x >> 64` wraps to a
    /// legal shift and the zero mask forces the result to 0 regardless of
    /// the container. Removing the `n == 0` branch saved ~2% of decode time
    /// (verified via callgrind).
    #[inline(always)]
    pub fn get_bits(&mut self, n: u32) -> u32 {
        debug_assert!(n <= 56);
        if self.bits_consumed as u32 + n > 64 {
            self.refill();
        }
        // SAFETY of shift: `self.bits_consumed <= 64` always (refill normalizes
        // it), and `n <= 56`, so `shift_by` ∈ [0, 64]. `x >> 64` is defined as
        // wrap-mod-bitwidth = `x >> 0` in release builds, i.e. `bit_container`,
        // but the mask is zero so the final `value` is still 0.
        let shift_by = 64u32.wrapping_sub(self.bits_consumed as u32).wrapping_sub(n);
        let mask = (1u64 << n).wrapping_sub(1);
        let value = ((self.bit_container.wrapping_shr(shift_by)) & mask) as u32;
        self.bits_consumed += n as u8;
        value
    }

    /// Ensure at least `n` bits are available in the container so a subsequent
    /// `peek_bits` is valid.  Used by Huffman decoders that peek wider than
    /// they consume.
    #[inline(always)]
    pub fn ensure_bits(&mut self, n: u32) {
        if self.bits_consumed as u32 + n > 64 {
            self.refill();
        }
    }

    /// Peek without consuming.  Assumes `ensure_bits(n)` was called first.
    #[inline(always)]
    pub fn peek_bits(&self, n: u32) -> u32 {
        debug_assert!(n > 0 && n <= 56);
        let shift_by = 64u32 - self.bits_consumed as u32 - n;
        let mask = (1u64 << n) - 1;
        ((self.bit_container >> shift_by) & mask) as u32
    }

    /// Consume `n` bits previously peeked.
    #[inline(always)]
    pub fn consume(&mut self, n: u32) {
        self.bits_consumed += n as u8;
    }

    /// Raw view for a caller-managed fast path: `(index, bits_consumed,
    /// container, source pointer)`. Only valid while the fast path keeps the
    /// window inside the source (it must check `index >= 8` before each
    /// refill; the stream-start phantom-bit state is never entered here
    /// because a refill from `index >= 8` always succeeds).
    #[inline(always)]
    pub fn raw_parts(&self) -> (usize, u32, u64, *const u8) {
        (self.index, self.bits_consumed as u32, self.bit_container, self.source.as_ptr())
    }

    /// True while the one-load refill fast path is still available (source
    /// >= 8 bytes and the stream start has not been reached).
    #[inline(always)]
    pub fn fast_ok(&self) -> bool {
        self.refill_limit != 0 && self.index + 8 <= self.source.len()
    }

    /// Write back state taken via [`raw_parts`].
    #[inline(always)]
    pub fn set_raw_parts(&mut self, index: usize, bits_consumed: u32, container: u64) {
        self.index = index;
        self.bits_consumed = bits_consumed as u8;
        self.bit_container = container;
    }

    /// Skip the trailing 0-padding and the 1-bit end-of-stream sentinel that
    /// terminates a zstd backward bitstream.  Errors if no sentinel is found in
    /// the first 8 bits (RFC 8878 limits padding to <8 bits).
    pub fn skip_padding_bits(&mut self) -> io::Result<()> {
        let mut skipped = 0;
        loop {
            let bit = self.get_bits(1);
            skipped += 1;
            if bit == 1 {
                return Ok(());
            }
            if skipped > 8 {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidData,
                    "zstd: missing end-of-stream sentinel in backward bitstream",
                ));
            }
        }
    }
}

/// Forward bit writer used to encode the zstd backward bitstream.
///
/// Bits are written LSB-first within each byte; bytes are emitted in increasing
/// address order.  The first bit written ends up at bit 0 of byte 0.  When the
/// decoder loads bytes from the END as a little-endian u64 and reads MSB-first,
/// it sees the LAST bit written first — so the encoder must emit symbols in
/// REVERSE order, then call `finalize` to add the end-of-stream sentinel.
pub struct ForwardBitWriter {
    pub output: Vec<u8>,
    /// Up to 64 unflushed bits, packed at the LSB side.
    partial: u64,
    bits_in_partial: u32,
}

impl ForwardBitWriter {
    pub fn new() -> Self {
        Self {
            output: Vec::new(),
            partial: 0,
            bits_in_partial: 0,
        }
    }


    /// Write the low `n` bits of `bits` (n ≤ 56).  Caller must ensure that
    /// the upper bits beyond `n` are zero.
    ///
    /// Drains BEFORE adding so the `bits << bits_in_partial` shift never
    /// overflows the u64.  Fast path drains 4 bytes; slow tail drains
    /// byte-by-byte if more room is still needed.
    #[inline]
    pub fn write_bits(&mut self, bits: u64, n: u32) {
        debug_assert!(n <= 56);
        debug_assert!(n == 64 || bits >> n == 0, "extra bits set above n");
        // Fast drain: 4 bytes at a time when bits_in_partial is high enough.
        if self.bits_in_partial + n > 64 && self.bits_in_partial >= 32 {
            self.output.reserve(4);
            unsafe {
                let len = self.output.len();
                let ptr = self.output.as_mut_ptr().add(len);
                let lo = self.partial as u32;
                std::ptr::write_unaligned(ptr as *mut u32, lo.to_le());
                self.output.set_len(len + 4);
            }
            self.partial >>= 32;
            self.bits_in_partial -= 32;
        }
        // Slow tail: if still not enough room, drain byte-by-byte.
        while self.bits_in_partial + n > 64 {
            self.output.push(self.partial as u8);
            self.partial >>= 8;
            self.bits_in_partial -= 8;
        }
        // `checked_shl` keeps the `n == 0` with 64 pending bits case defined.
        self.partial |= bits.checked_shl(self.bits_in_partial).unwrap_or(0);
        self.bits_in_partial += n;
    }

    /// Add the end-of-stream sentinel '1' bit and pad with zeroes to the next
    /// byte boundary.  After this call the buffer contains a complete zstd
    /// backward bitstream readable by `ReverseBitReader`.
    pub fn finalize(mut self) -> Vec<u8> {
        // Sentinel bit. If the partial buffer has 0 spare bits in the current byte,
        // we still need a 1 bit which lands at bit 0 of a fresh byte.
        self.write_bits(1, 1);
        // Flush any remaining partial bits to the next byte boundary.
        let leftover = self.bits_in_partial;
        if leftover > 0 {
            let bytes = ((leftover + 7) / 8) as usize;
            let buf = self.partial.to_le_bytes();
            self.output.extend_from_slice(&buf[..bytes]);
        }
        self.output
    }

    /// Pad with zeros to the next byte boundary and return the buffer.
    /// Used for FSE table descriptions, which are packed forward streams
    /// without the end-of-stream sentinel.
    pub fn finalize_no_sentinel(mut self) -> Vec<u8> {
        let leftover = self.bits_in_partial;
        if leftover > 0 {
            let bytes = ((leftover + 7) / 8) as usize;
            let buf = self.partial.to_le_bytes();
            self.output.extend_from_slice(&buf[..bytes]);
        }
        self.output
    }
}

impl Default for ForwardBitWriter {
    fn default() -> Self {
        Self::new()
    }
}

/// Simple forward byte reader for frame/block headers.
pub struct ForwardByteReader<'a> {
    pub data: &'a [u8],
    pub pos: usize,
}

impl<'a> ForwardByteReader<'a> {
    pub fn new(data: &'a [u8]) -> Self {
        Self { data, pos: 0 }
    }

    #[inline]
    pub fn position(&self) -> usize {
        self.pos
    }

    pub fn read_u8(&mut self) -> io::Result<u8> {
        if self.pos >= self.data.len() {
            return Err(io::Error::new(io::ErrorKind::UnexpectedEof, "zstd: unexpected end of frame"));
        }
        let v = self.data[self.pos];
        self.pos += 1;
        Ok(v)
    }

    pub fn read_u16_le(&mut self) -> io::Result<u16> {
        if self.pos + 2 > self.data.len() {
            return Err(io::Error::new(io::ErrorKind::UnexpectedEof, "zstd: unexpected end of frame"));
        }
        let v = u16::from_le_bytes([self.data[self.pos], self.data[self.pos + 1]]);
        self.pos += 2;
        Ok(v)
    }

    pub fn read_u24_le(&mut self) -> io::Result<u32> {
        if self.pos + 3 > self.data.len() {
            return Err(io::Error::new(io::ErrorKind::UnexpectedEof, "zstd: unexpected end of frame"));
        }
        let v = self.data[self.pos] as u32
            | (self.data[self.pos + 1] as u32) << 8
            | (self.data[self.pos + 2] as u32) << 16;
        self.pos += 3;
        Ok(v)
    }

    pub fn read_u32_le(&mut self) -> io::Result<u32> {
        if self.pos + 4 > self.data.len() {
            return Err(io::Error::new(io::ErrorKind::UnexpectedEof, "zstd: unexpected end of frame"));
        }
        let v = u32::from_le_bytes(self.data[self.pos..self.pos + 4].try_into().unwrap());
        self.pos += 4;
        Ok(v)
    }

    pub fn read_u64_le(&mut self) -> io::Result<u64> {
        if self.pos + 8 > self.data.len() {
            return Err(io::Error::new(io::ErrorKind::UnexpectedEof, "zstd: unexpected end of frame"));
        }
        let v = u64::from_le_bytes(self.data[self.pos..self.pos + 8].try_into().unwrap());
        self.pos += 8;
        Ok(v)
    }

    pub fn read_bytes(&mut self, n: usize) -> io::Result<&'a [u8]> {
        if self.pos + n > self.data.len() {
            return Err(io::Error::new(io::ErrorKind::UnexpectedEof, "zstd: unexpected end of frame"));
        }
        let slice = &self.data[self.pos..self.pos + n];
        self.pos += n;
        Ok(slice)
    }

    pub fn skip(&mut self, n: usize) -> io::Result<()> {
        if self.pos + n > self.data.len() {
            return Err(io::Error::new(io::ErrorKind::UnexpectedEof, "zstd: unexpected end of frame"));
        }
        self.pos += n;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reverse_reader_simple() {
        // Two bytes [0xAA, 0x55] = [0b10101010, 0b01010101].
        // Reading "backward MSB first": first bit comes from the top of byte 1 (0x55),
        // i.e. 0,1,0,1,0,1,0,1, then the bits of byte 0: 1,0,1,0,1,0,1,0.
        let data = [0xAA, 0x55];
        let mut br = ReverseBitReader::new(&data).unwrap();
        // ruzstd's matching test case (no skip_padding_bits because we want the raw stream):
        assert_eq!(br.get_bits(1), 0);
        assert_eq!(br.get_bits(1), 1);
        assert_eq!(br.get_bits(1), 0);
        assert_eq!(br.get_bits(4), 0b1010);
        assert_eq!(br.get_bits(4), 0b1101);
        assert_eq!(br.get_bits(4), 0b0101);
        // After 16 bits, anything more is zeros and bits_remaining goes negative.
        assert_eq!(br.get_bits(4), 0b0000);
        assert!(br.bits_remaining() < 0);
    }

    #[test]
    fn skip_padding_with_sentinel() {
        // Last byte 0x80 = 0b10000000: sentinel at bit 7, 0 padding bits before it.
        // Data is bits 6..0 of last byte then all of byte 0 (0xAB).
        let data = [0xAB, 0x80];
        let mut br = ReverseBitReader::new(&data).unwrap();
        br.skip_padding_bits().unwrap();
        // After skip, total data bits = 7 + 8 = 15 remaining.
        assert_eq!(br.bits_remaining(), 15);
        // Read the 7 low bits of 0x80: all zero.
        assert_eq!(br.get_bits(7), 0);
        // Then byte 0xAB = 0b10101011 in MSB-first order.
        assert_eq!(br.get_bits(8), 0xAB);
        assert_eq!(br.bits_remaining(), 0);
    }

    #[test]
    fn skip_padding_short_sentinel() {
        // Last byte 0x01: 7 padding zeros, sentinel at bit 0. No data in last byte.
        let data = [0xAB, 0x01];
        let mut br = ReverseBitReader::new(&data).unwrap();
        br.skip_padding_bits().unwrap();
        assert_eq!(br.bits_remaining(), 8);
        assert_eq!(br.get_bits(8), 0xAB);
    }

    #[test]
    fn errors_on_zero_last_byte() {
        let data = [0x00];
        let mut br = ReverseBitReader::new(&data).unwrap();
        assert!(br.skip_padding_bits().is_err());
    }

    /// Encoder writes symbols in REVERSE; the reverse-MSB decoder must
    /// recover them in FORWARD order.
    #[test]
    fn writer_reader_roundtrip_simple() {
        // Encode the sequence [3, 7, 1, 0] with 4 bits each, then decode
        // and expect to read them in forward order.
        let symbols: [u32; 4] = [3, 7, 1, 0];
        let mut bw = ForwardBitWriter::new();
        for &s in symbols.iter().rev() {
            bw.write_bits(s as u64, 4);
        }
        let bytes = bw.finalize();

        let mut br = ReverseBitReader::new(&bytes).unwrap();
        br.skip_padding_bits().unwrap();
        for &expected in &symbols {
            assert_eq!(br.get_bits(4) as u32, expected);
        }
    }

    #[test]
    fn writer_reader_roundtrip_mixed_widths() {
        // Encoder side, in DECODE order: read 5 bits = 0x1A, then 3 = 0b101,
        // then 12 = 0xABC, then 8 = 0xFF.
        let plan: &[(u64, u32)] = &[(0x1A, 5), (0b101, 3), (0xABC, 12), (0xFF, 8)];
        let mut bw = ForwardBitWriter::new();
        for &(v, n) in plan.iter().rev() {
            bw.write_bits(v, n);
        }
        let bytes = bw.finalize();

        let mut br = ReverseBitReader::new(&bytes).unwrap();
        br.skip_padding_bits().unwrap();
        for &(v, n) in plan {
            assert_eq!(br.get_bits(n) as u64, v);
        }
    }
}
