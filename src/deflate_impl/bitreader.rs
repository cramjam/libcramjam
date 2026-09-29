//! Bit-level reader for DEFLATE decompression.
//! Bits are packed LSB-first into bytes (RFC 1951 section 3.1.1).
//!
//! Shape follows libdeflate / zlib-rs: a 64-bit accumulator refilled with a
//! single unaligned 8-byte load (branchless, always lands on 56..=63 valid
//! bits) while at least 8 input bytes remain, with a cold byte-at-a-time
//! tail near the end of input. Past the end of input the tail feeds zero
//! bytes and counts them in `overrun`; consuming any of those zero bits is
//! how truncation is detected.

use std::io;

pub struct BitReader<'a> {
    data: &'a [u8],
    /// Next input byte to load.
    pos: usize,
    bit_buf: u64,
    /// Valid bits in `bit_buf`. Invariant: `bit_count <= 63`.
    bit_count: u32,
    /// Zero bytes "loaded" past the end of `data`.
    overrun: u32,
}

#[cold]
#[inline(never)]
fn truncated() -> io::Error {
    io::Error::new(io::ErrorKind::UnexpectedEof, "unexpected end of deflate stream")
}

impl<'a> BitReader<'a> {
    pub fn new(data: &'a [u8]) -> Self {
        Self {
            data,
            pos: 0,
            bit_buf: 0,
            bit_count: 0,
            overrun: 0,
        }
    }

    /// Top the accumulator up to at least 56 valid bits. Returns `false`
    /// only when the stream is exhausted *and* previously supplied padding
    /// bits have already been consumed (i.e. the input is truncated).
    #[inline(always)]
    pub fn refill(&mut self) -> bool {
        if self.pos + 8 <= self.data.len() {
            // SAFETY: 8 bytes available at `pos`.
            let v = unsafe {
                u64::from_le(std::ptr::read_unaligned(self.data.as_ptr().add(self.pos) as *const u64))
            };
            // bit_count <= 63 so the shift is in range; whole bytes that fit
            // are consumed and the count lands in 56..=63.
            self.bit_buf |= v << self.bit_count;
            self.pos += ((63 - self.bit_count) >> 3) as usize;
            self.bit_count |= 56;
            return true;
        }
        self.refill_slow()
    }

    #[cold]
    #[inline(never)]
    fn refill_slow(&mut self) -> bool {
        if self.overrun * 8 > self.bit_count {
            // Already consumed bits that were never in the input.
            return false;
        }
        while self.bit_count < 56 {
            if self.pos < self.data.len() {
                self.bit_buf |= (self.data[self.pos] as u64) << self.bit_count;
                self.pos += 1;
            } else {
                self.overrun += 1;
            }
            self.bit_count += 8;
        }
        true
    }

    /// `true` if any padding (past-the-end) bits have been consumed.
    #[inline(always)]
    pub fn overread(&self) -> bool {
        self.overrun * 8 > self.bit_count
    }

    /// Peek the low `n` bits (`n <= 32`). Caller must have refilled.
    #[inline(always)]
    pub fn peek(&self, n: u32) -> u32 {
        (self.bit_buf & ((1u64 << n) - 1)) as u32
    }

    /// Consume `n` bits (`n <= bit_count`).
    #[inline(always)]
    pub fn consume(&mut self, n: u32) {
        self.bit_buf >>= n;
        self.bit_count -= n;
    }

    /// Read `n` bits (`0 <= n <= 32`) without a refill check — caller
    /// guarantees the accumulator holds them.
    #[inline(always)]
    pub fn bits(&mut self, n: u32) -> u32 {
        let v = self.peek(n);
        self.consume(n);
        v
    }

    /// Checked read of `n` bits (0..=32) for header parsing.
    #[inline]
    pub fn read_bits(&mut self, n: u32) -> io::Result<u32> {
        if self.bit_count < n {
            if !self.refill() || self.bit_count < n {
                return Err(truncated());
            }
        }
        let v = self.bits(n);
        if self.overread() {
            return Err(truncated());
        }
        Ok(v)
    }

    /// Discard bits until aligned to a byte boundary.
    pub fn align_to_byte(&mut self) {
        let discard = self.bit_count & 7;
        self.bit_buf >>= discard;
        self.bit_count -= discard;
    }

    /// Read raw bytes (aligns to byte boundary first, then reads from buffer/data).
    pub fn read_bytes(&mut self, n: usize) -> io::Result<&'a [u8]> {
        self.align_to_byte();
        let buffered = (self.bit_count / 8) as isize - self.overrun as isize;
        if buffered < 0 {
            return Err(truncated());
        }
        let start = self.pos - buffered as usize;
        if start + n > self.data.len() {
            return Err(truncated());
        }
        self.bit_buf = 0;
        self.bit_count = 0;
        self.overrun = 0;
        self.pos = start + n;
        Ok(&self.data[start..start + n])
    }

    /// Number of input bytes consumed so far (rounded up to byte boundary).
    pub fn bytes_consumed(&self) -> usize {
        let buffered = (self.bit_count / 8) as isize - self.overrun as isize;
        self.pos - buffered.max(0) as usize
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
        assert!(r.read_bits(1).is_err());
        assert_eq!(r.bytes_consumed(), 1);
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
        assert_eq!(r.bytes_consumed(), 5);
    }

    #[test]
    fn test_bytes_consumed_long() {
        let data: Vec<u8> = (0..40u8).collect();
        let mut r = BitReader::new(&data);
        for _ in 0..10 {
            r.read_bits(13).unwrap();
        }
        // 130 bits consumed -> 17 bytes (rounded up)
        assert_eq!(r.bytes_consumed(), 17);
        assert_eq!(r.read_bytes(2).unwrap(), &[17, 18]);
    }
}
