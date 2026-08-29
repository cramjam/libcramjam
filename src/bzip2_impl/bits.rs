//! Bit-level I/O for bzip2.
//!
//! bzip2 packs bits MSB-first within each byte and processes bytes in input
//! order — the opposite of DEFLATE / zstd.  A 5-bit field whose decimal value
//! is `0b10110` lands in bits 7..3 of byte 0 (high bits to low bits).
//!
//! The reader exposes `read_bits(n)` (n ≤ 56) which lazily refills a 64-bit
//! container.  The writer mirrors it with `write_bits(value, n)`.

use std::io;

// =========================================================================
// Reader
// =========================================================================

/// MSB-first big-endian bit reader.  Wraps a borrowed byte slice.
pub struct BitReader<'a> {
    data: &'a [u8],
    pos: usize,
    /// Up to 64 unread bits, packed MSB-first (the next bit to return is
    /// at the highest set position in the container).
    container: u64,
    /// Number of valid bits in `container`.
    bits_in_container: u32,
}

impl<'a> BitReader<'a> {
    pub fn new(data: &'a [u8]) -> Self {
        Self {
            data,
            pos: 0,
            container: 0,
            bits_in_container: 0,
        }
    }

    /// Total bits read since construction (used for diagnostics).
    pub fn bit_position(&self) -> usize {
        self.pos * 8 - self.bits_in_container as usize
    }

    /// Read `n` bits (1..=32) MSB-first as a u32.
    #[inline]
    pub fn read_bits(&mut self, n: u32) -> io::Result<u32> {
        debug_assert!(n >= 1 && n <= 32);
        Ok(self.read_bits_u64(n)? as u32)
    }

    /// Read `n` bits (1..=56) MSB-first as a u64.  Used for fields larger
    /// than 32 bits (e.g. the 48-bit block magic).
    #[inline]
    pub fn read_bits_u64(&mut self, n: u32) -> io::Result<u64> {
        debug_assert!(n >= 1 && n <= 56);
        while self.bits_in_container < n {
            if self.pos >= self.data.len() {
                return Err(io::Error::new(
                    io::ErrorKind::UnexpectedEof,
                    "bzip2: ran out of bits",
                ));
            }
            self.container = (self.container << 8) | self.data[self.pos] as u64;
            self.pos += 1;
            self.bits_in_container += 8;
        }
        let shift = self.bits_in_container - n;
        let mask: u64 = (1u64 << n) - 1;
        let value = (self.container >> shift) & mask;
        // Mask out the bits we just consumed.
        self.container &= (1u64 << shift).wrapping_sub(1);
        self.bits_in_container -= n;
        Ok(value)
    }

    /// Read a single bit as a bool.
    #[inline]
    pub fn read_bit(&mut self) -> io::Result<bool> {
        Ok(self.read_bits(1)? != 0)
    }

    /// Refill the container so it has at least `n` bits queued.  Reads up to
    /// 7 bytes from the underlying slice if needed; if the slice is exhausted
    /// before we hit `n` bits, returns UnexpectedEof.
    #[inline]
    pub fn refill(&mut self, n: u32) -> io::Result<()> {
        debug_assert!(n <= 56);
        while self.bits_in_container < n {
            if self.pos >= self.data.len() {
                return Err(io::Error::new(
                    io::ErrorKind::UnexpectedEof,
                    "bzip2: ran out of bits",
                ));
            }
            self.container = (self.container << 8) | self.data[self.pos] as u64;
            self.pos += 1;
            self.bits_in_container += 8;
        }
        Ok(())
    }

    /// Peek the top `n` bits of the container WITHOUT consuming them.  Caller
    /// must have already called `refill(n)` (or another method that ensured
    /// the container has at least `n` bits).
    #[inline]
    pub fn peek(&self, n: u32) -> u32 {
        debug_assert!(self.bits_in_container >= n);
        let shift = self.bits_in_container - n;
        let mask: u64 = (1u64 << n) - 1;
        ((self.container >> shift) & mask) as u32
    }

    /// Drop the top `n` bits of the container.  Caller is responsible for
    /// ensuring `bits_in_container >= n` before calling.
    #[inline]
    pub fn consume(&mut self, n: u32) {
        debug_assert!(self.bits_in_container >= n);
        let shift = self.bits_in_container - n;
        self.container &= (1u64 << shift).wrapping_sub(1);
        self.bits_in_container -= n;
    }

    /// Skip remaining bits in the current byte and return how many were
    /// skipped (0..=7).  Used after reading bit-aligned data to land back
    /// on a byte boundary.
    pub fn align_to_byte(&mut self) {
        let extra = self.bits_in_container % 8;
        self.bits_in_container -= extra;
        self.container &= (1u64 << self.bits_in_container).wrapping_sub(1);
    }

    /// Number of complete bytes still available in the underlying slice
    /// after the container's queued bits.  Useful for diagnostics.
    pub fn bytes_remaining(&self) -> usize {
        self.data.len() - self.pos
    }
}

// =========================================================================
// Writer
// =========================================================================

/// MSB-first big-endian bit writer.  Bits are accumulated in a 64-bit
/// container and flushed to bytes once at least 8 bits are queued.
pub struct BitWriter {
    pub output: Vec<u8>,
    container: u64,
    bits_in_container: u32,
}

impl BitWriter {
    pub fn new() -> Self {
        Self {
            output: Vec::new(),
            container: 0,
            bits_in_container: 0,
        }
    }

    pub fn with_capacity(cap: usize) -> Self {
        Self {
            output: Vec::with_capacity(cap),
            container: 0,
            bits_in_container: 0,
        }
    }

    /// Write the low `n` bits of `value` MSB-first (1..=56).
    #[inline]
    pub fn write_bits(&mut self, value: u64, n: u32) {
        debug_assert!(n <= 56);
        debug_assert!(n == 64 || value >> n == 0, "extra bits set above n");
        // Invariant between calls: `bits_in_container < 32` and the
        // container holds exactly those bits (upper bits clear).  Fields
        // wider than 32 bits (the 48-bit magics) are split so the shift
        // below never overflows (< 32 + 32 = 64 bits).
        if n > 32 {
            self.write_bits(value >> 32, n - 32);
            self.write_bits(value & 0xFFFF_FFFF, 32);
            return;
        }
        self.container = (self.container << n) | value;
        self.bits_in_container += n;
        // Drain 4 bytes at a time while >= 32 bits are queued: one iteration
        // for typical Huffman fields, two for the rare wide field.
        while self.bits_in_container >= 32 {
            self.bits_in_container -= 32;
            let chunk = ((self.container >> self.bits_in_container) as u32).to_be_bytes();
            self.output.extend_from_slice(&chunk);
        }
        // `bits_in_container < 32`, so the shift is in range (0 → mask 0).
        self.container &= (1u64 << self.bits_in_container) - 1;
    }

    /// Pad to the next byte boundary with zero bits.
    pub fn align_to_byte(&mut self) {
        let extra = self.bits_in_container % 8;
        if extra > 0 {
            self.write_bits(0, 8 - extra);
        }
        // Flush the (now whole) queued bytes, MSB-first.
        while self.bits_in_container >= 8 {
            self.bits_in_container -= 8;
            self.output.push((self.container >> self.bits_in_container) as u8);
        }
        self.container = 0;
    }

    /// Drop the writer and return the encoded bytes.  Caller must have
    /// already byte-aligned the stream.
    pub fn finish(mut self) -> Vec<u8> {
        debug_assert_eq!(self.bits_in_container, 0, "BitWriter::finish called mid-byte");
        // Defensive: pad if we somehow weren't aligned.
        if self.bits_in_container > 0 {
            self.align_to_byte();
        }
        std::mem::take(&mut self.output)
    }
}

impl Default for BitWriter {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn write_then_read_simple() {
        let mut bw = BitWriter::new();
        bw.write_bits(0b101, 3);
        bw.write_bits(0b1100, 4);
        bw.write_bits(0b1, 1);
        // Total 8 bits → one byte = 0b1011_1001 = 0xB9
        let bytes = bw.finish();
        assert_eq!(bytes, vec![0xB9]);

        let mut br = BitReader::new(&bytes);
        assert_eq!(br.read_bits(3).unwrap(), 0b101);
        assert_eq!(br.read_bits(4).unwrap(), 0b1100);
        assert_eq!(br.read_bits(1).unwrap(), 0b1);
    }

    #[test]
    fn multi_byte_field() {
        let mut bw = BitWriter::new();
        bw.write_bits(0xDEADBEEF, 32);
        bw.write_bits(0x12, 8);
        let bytes = bw.finish();
        assert_eq!(bytes, vec![0xDE, 0xAD, 0xBE, 0xEF, 0x12]);

        let mut br = BitReader::new(&bytes);
        assert_eq!(br.read_bits(32).unwrap(), 0xDEADBEEF);
        assert_eq!(br.read_bits(8).unwrap(), 0x12);
    }

    #[test]
    fn cross_byte_field() {
        // 12-bit field straddling a byte boundary.
        let mut bw = BitWriter::new();
        bw.write_bits(0b0101_1100_1010, 12);
        bw.write_bits(0b1111, 4);
        let bytes = bw.finish();
        // 12 + 4 = 16 bits = 2 bytes.  MSB first: 0101_1100 1010_1111 = 0x5C 0xAF
        assert_eq!(bytes, vec![0x5C, 0xAF]);

        let mut br = BitReader::new(&bytes);
        assert_eq!(br.read_bits(12).unwrap(), 0b0101_1100_1010);
        assert_eq!(br.read_bits(4).unwrap(), 0b1111);
    }

    #[test]
    fn read_eof() {
        let mut br = BitReader::new(&[0xFF]);
        assert!(br.read_bits(8).is_ok());
        assert!(br.read_bits(1).is_err());
    }
}
