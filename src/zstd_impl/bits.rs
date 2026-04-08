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
        })
    }

    /// Bits remaining (signed).  Negative once the reader has overrun the input.
    #[inline]
    pub fn bits_remaining(&self) -> isize {
        self.index as isize * 8 + (64 - self.bits_consumed as isize) - self.extra_bits as isize
    }

    /// Refill the container so at least 56 bits are available (when possible).
    #[cold]
    fn refill(&mut self) {
        let bytes_consumed = (self.bits_consumed / 8) as usize;
        if bytes_consumed == 0 {
            return;
        }

        if self.index >= bytes_consumed {
            // Move the window down by `bytes_consumed`.
            self.index -= bytes_consumed;
            self.bits_consumed &= 7;
            // Read 8 bytes ending at index+8 from the source.
            let end = self.index + 8;
            if end <= self.source.len() {
                self.bit_container = u64::from_le_bytes(
                    self.source[self.index..end].try_into().unwrap(),
                );
            } else {
                // Near the start: read whatever is available, zero-pad.
                let mut buf = [0u8; 8];
                let avail = self.source.len() - self.index;
                buf[..avail].copy_from_slice(&self.source[self.index..]);
                self.bit_container = u64::from_le_bytes(buf);
            }
        } else if self.index > 0 {
            // Last partial load: read from offset 0.
            if self.source.len() >= 8 {
                self.bit_container =
                    u64::from_le_bytes((&self.source[..8]).try_into().unwrap());
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
        } else if self.bits_consumed < 64 {
            // index == 0 but partial bits remain.
            self.bit_container <<= self.bits_consumed;
            self.extra_bits += self.bits_consumed as usize;
            self.bits_consumed = 0;
        } else {
            // Fully exhausted — return zeros.
            self.extra_bits += self.bits_consumed as usize;
            self.bits_consumed = 0;
            self.bit_container = 0;
        }
    }

    /// Read up to 56 bits.  Reading more than the stream contains returns zero
    /// bits but advances `bits_remaining()` into negative territory.
    #[inline]
    pub fn get_bits(&mut self, n: u32) -> u32 {
        if n == 0 {
            return 0;
        }
        if self.bits_consumed as u32 + n > 64 {
            self.refill();
        }
        let value = self.peek_bits(n);
        self.consume(n);
        value
    }

    /// Ensure at least `n` bits are available in the container so a subsequent
    /// `peek_bits` is valid.  Used by Huffman decoders that peek wider than
    /// they consume.
    #[inline]
    pub fn ensure_bits(&mut self, n: u32) {
        if self.bits_consumed as u32 + n > 64 {
            self.refill();
        }
    }

    /// Peek without consuming.
    #[inline]
    pub fn peek_bits(&self, n: u32) -> u32 {
        if n == 0 {
            return 0;
        }
        let shift_by = 64u32 - self.bits_consumed as u32 - n;
        let mask = if n >= 32 { u32::MAX as u64 } else { (1u64 << n) - 1 };
        ((self.bit_container >> shift_by) & mask) as u32
    }

    /// Consume `n` bits previously peeked.
    #[inline]
    pub fn consume(&mut self, n: u32) {
        self.bits_consumed += n as u8;
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
    pub fn remaining(&self) -> usize {
        self.data.len() - self.pos
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
}
