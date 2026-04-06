//! Bit-level readers for Zstandard.
//!
//! Zstd uses **backward** bitstreams: data is read from the last byte toward
//! the first, MSB-first within each loaded word.  This is the opposite of
//! DEFLATE's LSB-first forward streams.

use std::io;

/// Read little-endian u64 from a byte slice (with bounds padding).
#[inline(always)]
fn read_le_u64(data: &[u8], pos: usize) -> u64 {
    // Read up to 8 bytes ending at `pos` (exclusive).
    let start = pos.saturating_sub(8);
    let slice = &data[start..pos];
    let mut buf = [0u8; 8];
    buf[..slice.len()].copy_from_slice(slice);
    u64::from_le_bytes(buf)
}

/// Backward bit reader for FSE and Huffman decoding.
///
/// Models the zstd `BIT_DStream_t`: a 64-bit container loaded little-endian
/// from a pointer that moves backward through the data.
pub struct ReverseBitReader<'a> {
    data: &'a [u8],
    /// Index into `data`: the next load reads bytes ending here.
    ptr: usize,
    /// 64-bit accumulator (bits read from MSB side).
    container: u64,
    /// Number of bits consumed from `container` (0 = fresh, 64 = empty).
    consumed: u32,
    /// Total data bits in the stream (set once at init, never changes).
    total_bits: u32,
    /// Total data bits consumed so far.
    bits_read: u32,
}

impl<'a> ReverseBitReader<'a> {
    /// Initialize from a byte slice.  Finds the sentinel bit in the last byte.
    pub fn new(data: &'a [u8]) -> io::Result<Self> {
        if data.is_empty() {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "zstd: empty bitstream",
            ));
        }

        let last = *data.last().unwrap();
        if last == 0 {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "zstd: bitstream last byte is zero (no sentinel)",
            ));
        }

        let ptr = data.len();
        let container = read_le_u64(data, ptr);
        let loaded = ptr.min(8);
        let consumed = ((8 - loaded) * 8) as u32 + 1 + last.leading_zeros();

        // Total data bits = (N-1)*8 + (7 - leading_zeros)
        let total_bits = if data.len() == 1 {
            7u32.saturating_sub(last.leading_zeros())
        } else {
            (data.len() as u32 - 1) * 8 + 7 - last.leading_zeros()
        };

        Ok(Self {
            data,
            ptr,
            container,
            consumed,
            total_bits,
            bits_read: 0,
        })
    }

    /// Peek at the next `n` bits (0..=32) without consuming them.
    #[inline(always)]
    pub fn peek_bits(&self, n: u32) -> u32 {
        if n == 0 { return 0; }
        let shifted = if self.consumed < 64 {
            self.container << self.consumed
        } else {
            0
        };
        (shifted >> (64 - n)) as u32
    }

    /// Consume `n` bits.
    #[inline(always)]
    pub fn consume(&mut self, n: u32) {
        self.consumed += n;
        self.bits_read += n;
    }

    /// Read `n` bits (peek + consume).
    #[inline(always)]
    pub fn read_bits(&mut self, n: u32) -> u32 {
        let val = self.peek_bits(n);
        self.consume(n);
        val
    }

    /// Reload the container from the data stream.  Call this periodically
    /// to ensure enough bits are available for the next read.
    #[inline(always)]
    pub fn reload(&mut self) {
        let bytes_consumed = (self.consumed >> 3) as usize;
        if bytes_consumed == 0 {
            return;
        }
        // Don't reload if all remaining data is already in the current container.
        if self.ptr <= bytes_consumed {
            return;
        }
        self.ptr -= bytes_consumed;
        self.consumed &= 7;
        self.container = read_le_u64(self.data, self.ptr);
    }

    /// Check if all data bits have been consumed.
    #[inline]
    pub fn is_done(&self) -> bool {
        self.bits_read >= self.total_bits
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
