//! Bit-level reader for DEFLATE decompression.
//! Bits are packed LSB-first into bytes (RFC 1951 section 3.1.1).

use std::io;

pub struct BitReader<'a> {
    data: &'a [u8],
    pos: usize,
    bit_buf: u64,
    bit_count: u32,
}

impl<'a> BitReader<'a> {
    pub fn new(data: &'a [u8]) -> Self {
        Self {
            data,
            pos: 0,
            bit_buf: 0,
            bit_count: 0,
        }
    }

    #[inline]
    fn fill(&mut self, need: u32) {
        while self.bit_count < need && self.pos < self.data.len() {
            self.bit_buf |= (self.data[self.pos] as u64) << self.bit_count;
            self.pos += 1;
            self.bit_count += 8;
        }
    }

    /// Read n bits (0..=57) as a u32, LSB first.
    #[inline]
    pub fn read_bits(&mut self, n: u32) -> io::Result<u32> {
        if n == 0 {
            return Ok(0);
        }
        self.fill(n);
        if self.bit_count < n {
            return Err(io::Error::new(
                io::ErrorKind::UnexpectedEof,
                "unexpected end of deflate stream",
            ));
        }
        let mask = (1u64 << n) - 1;
        let value = (self.bit_buf & mask) as u32;
        self.bit_buf >>= n;
        self.bit_count -= n;
        Ok(value)
    }

    /// Peek at the next n bits without consuming them.
    ///
    /// If fewer than n bits are available, the result is zero-padded in the
    /// upper bits.  This is safe for Huffman table lookups because short codes
    /// are replicated across all suffix extensions.  Returns `Err` only when
    /// **zero** bits are available.
    #[inline]
    pub fn peek_bits(&mut self, n: u32) -> io::Result<u32> {
        self.fill(n);
        if self.bit_count == 0 {
            return Err(io::Error::new(
                io::ErrorKind::UnexpectedEof,
                "unexpected end of deflate stream",
            ));
        }
        let avail = std::cmp::min(self.bit_count, n);
        Ok((self.bit_buf & ((1u64 << avail) - 1)) as u32)
    }

    /// Consume n bits (after a successful peek).
    #[inline]
    pub fn consume(&mut self, n: u32) {
        self.bit_buf >>= n;
        self.bit_count -= n;
    }

    /// Discard bits until aligned to a byte boundary.
    pub fn align_to_byte(&mut self) {
        let discard = self.bit_count % 8;
        if discard > 0 {
            self.bit_buf >>= discard;
            self.bit_count -= discard;
        }
    }

    /// Read raw bytes (aligns to byte boundary first, then reads from buffer/data).
    pub fn read_bytes(&mut self, n: usize) -> io::Result<&'a [u8]> {
        self.align_to_byte();
        let buffered = (self.bit_count / 8) as usize;
        let start = self.pos - buffered;
        if start + n > self.data.len() {
            return Err(io::Error::new(
                io::ErrorKind::UnexpectedEof,
                "unexpected end of deflate stream",
            ));
        }
        self.bit_buf = 0;
        self.bit_count = 0;
        self.pos = start + n;
        Ok(&self.data[start..start + n])
    }

    /// Number of input bytes consumed so far (rounded up to byte boundary).
    pub fn bytes_consumed(&self) -> usize {
        let total_loaded_bits = self.pos * 8;
        let consumed_bits = total_loaded_bits - self.bit_count as usize;
        (consumed_bits + 7) / 8
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_read_bits() {
        // 0xA5 = 10100101, LSB-first: bit0=1,1=0,2=1,3=0,4=0,5=1,6=0,7=1
        let data = [0xA5u8];
        let mut r = BitReader::new(&data);
        assert_eq!(r.read_bits(1).unwrap(), 1); // bit0 = 1
        assert_eq!(r.read_bits(1).unwrap(), 0); // bit1 = 0
        assert_eq!(r.read_bits(3).unwrap(), 0b001); // bits 2-4 = 1,0,0 → 1
        assert_eq!(r.read_bits(3).unwrap(), 0b101); // bits 5-7 = 1,0,1 → 5
    }

    #[test]
    fn test_read_bytes() {
        let data = [0xFF, 0x01, 0x02, 0x03, 0x04];
        let mut r = BitReader::new(&data);
        // Read 3 bits from first byte
        assert_eq!(r.read_bits(3).unwrap(), 0b111);
        // Align and read 4 bytes
        let bytes = r.read_bytes(4).unwrap();
        assert_eq!(bytes, &[0x01, 0x02, 0x03, 0x04]);
    }
}
