//! Bit-level writer for DEFLATE compression.
//! Bits are packed LSB-first into bytes (RFC 1951 section 3.1.1).
//!
//! 64-bit accumulator; bytes are drained four at a time so the per-symbol
//! cost is one shift/or plus a predictable branch.

pub struct BitWriter {
    buf: Vec<u8>,
    bit_buf: u64,
    bit_count: u32,
}

impl BitWriter {
    pub fn with_capacity(cap: usize) -> Self {
        Self {
            buf: Vec::with_capacity(cap),
            bit_buf: 0,
            bit_count: 0,
        }
    }

    /// Write `n` bits (0..=32) from `value`, LSB first. `value` must have
    /// no bits set above `n`.
    #[inline(always)]
    pub fn write_bits(&mut self, value: u32, n: u32) {
        debug_assert!(n <= 32 && (n == 32 || value >> n == 0));
        self.bit_buf |= (value as u64) << self.bit_count;
        self.bit_count += n;
        if self.bit_count >= 32 {
            self.buf.extend_from_slice(&(self.bit_buf as u32).to_le_bytes());
            self.bit_buf >>= 32;
            self.bit_count -= 32;
        }
    }

    /// Pad remaining bits to the next byte boundary with zeros.
    pub fn align_to_byte(&mut self) {
        while self.bit_count > 0 {
            self.buf.push((self.bit_buf & 0xFF) as u8);
            self.bit_buf >>= 8;
            self.bit_count = self.bit_count.saturating_sub(8);
        }
        self.bit_buf = 0;
    }

    /// Write a raw byte (must be byte-aligned).
    pub fn write_byte(&mut self, byte: u8) {
        debug_assert_eq!(self.bit_count, 0);
        self.buf.push(byte);
    }

    /// Write a 16-bit value in little-endian order (must be byte-aligned).
    pub fn write_u16_le(&mut self, value: u16) {
        self.write_byte(value as u8);
        self.write_byte((value >> 8) as u8);
    }

    /// Write raw bytes (must be byte-aligned).
    pub fn write_bytes(&mut self, bytes: &[u8]) {
        debug_assert_eq!(self.bit_count, 0);
        self.buf.extend_from_slice(bytes);
    }

    /// Take the whole bytes emitted so far (bits in the accumulator stay).
    pub fn take_bytes(&mut self) -> Vec<u8> {
        std::mem::take(&mut self.buf)
    }

    /// Whole bytes currently buffered.
    pub fn buffered_len(&self) -> usize {
        self.buf.len()
    }

    /// Consume the writer and return the output buffer.
    pub fn finish(mut self) -> Vec<u8> {
        self.align_to_byte();
        self.buf
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_write_bits() {
        let mut w = BitWriter::with_capacity(8);
        w.write_bits(0b101, 3); // bits 0-2: 1,0,1
        w.write_bits(0b11, 2); // bits 3-4: 1,1
        w.write_bits(0b010, 3); // bits 5-7: 0,1,0
        let out = w.finish();
        // byte = bit0..7 = 1,0,1,1,1,0,1,0 = 0b01011101 = 0x5D
        assert_eq!(out, vec![0x5D]);
    }

    #[test]
    fn test_write_u16_le() {
        let mut w = BitWriter::with_capacity(2);
        w.write_u16_le(0x0201);
        let out = w.finish();
        assert_eq!(out, vec![0x01, 0x02]);
    }

    #[test]
    fn test_many_bits_cross_word_boundary() {
        let mut w = BitWriter::with_capacity(64);
        for i in 0..100u32 {
            w.write_bits(i & 0x1FFF, 13);
        }
        w.write_bits(0, 5);
        let out = w.finish();
        let mut acc = 0u64;
        let mut n = 0u32;
        let mut it = out.iter();
        for i in 0..100u32 {
            while n < 13 {
                acc |= (*it.next().unwrap() as u64) << n;
                n += 8;
            }
            assert_eq!((acc & 0x1FFF) as u32, i & 0x1FFF);
            acc >>= 13;
            n -= 13;
        }
    }
}
