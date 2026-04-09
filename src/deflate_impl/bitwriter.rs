//! Bit-level writer for DEFLATE compression.
//! Bits are packed LSB-first into bytes (RFC 1951 section 3.1.1).

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

    /// Write n bits (0..=32) from value, LSB first.
    #[inline]
    pub fn write_bits(&mut self, value: u32, n: u32) {
        self.bit_buf |= (value as u64) << self.bit_count;
        self.bit_count += n;
        self.flush_bytes();
    }

    #[inline]
    fn flush_bytes(&mut self) {
        while self.bit_count >= 8 {
            self.buf.push((self.bit_buf & 0xFF) as u8);
            self.bit_buf >>= 8;
            self.bit_count -= 8;
        }
    }

    /// Pad remaining bits to the next byte boundary with zeros.
    pub fn align_to_byte(&mut self) {
        if self.bit_count > 0 {
            self.buf.push((self.bit_buf & 0xFF) as u8);
            self.bit_buf = 0;
            self.bit_count = 0;
        }
    }

    /// Write a raw byte (flushes any partial byte first).
    pub fn write_byte(&mut self, byte: u8) {
        self.write_bits(byte as u32, 8);
    }

    /// Write a 16-bit value in little-endian order.
    pub fn write_u16_le(&mut self, value: u16) {
        self.write_byte(value as u8);
        self.write_byte((value >> 8) as u8);
    }

    /// Write raw bytes.
    pub fn write_bytes(&mut self, bytes: &[u8]) {
        for &b in bytes {
            self.write_byte(b);
        }
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
}
