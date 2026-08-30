//! lz4 de/compression interface — pure Rust frame & block.

use std::io::{Error, Read, Write};

/// Default compression level for LZ4 HC mode, matching the previously used
/// C `lz4` crate's `EncoderBuilder` default of 4.  Levels 0-2 use the fast
/// hash-table parser; levels 3-12 use the HC (High Compression) parser.
/// Passing `None` to `compress()` uses fast mode (level 0).
pub const DEFAULT_COMPRESSION_LEVEL: u32 = 4;
pub const LZ4_ACCELERATION_MAX: u32 = 65537;

pub use crate::lz4_impl::Lz4StreamCompressor;

/// Decompress lz4 frame data.
#[inline(always)]
pub fn decompress<W: Write + ?Sized, R: Read>(input: R, output: &mut W) -> Result<usize, Error> {
    crate::lz4_impl::decompress(input, output)
}

/// Worst-case compressed size of `input_len` bytes.
#[inline(always)]
pub fn compress_bound(input_len: usize, level: Option<u32>) -> usize {
    crate::lz4_impl::compress_bound(input_len, level)
}

/// Compress as an lz4 frame.
#[inline(always)]
pub fn compress<W: Write + ?Sized, R: Read>(
    input: R,
    output: &mut W,
    level: Option<u32>,
) -> Result<usize, Error> {
    crate::lz4_impl::compress(input, output, level)
}

/// Block-format helpers (no frame wrapper).
pub mod block {
    use std::io::Error;

    /// Worst-case compressed size for `n` bytes.  When `prepend_size` is true
    /// the output also has a 4-byte length prefix that the matching
    /// `decompress_vec` will read.
    #[inline(always)]
    pub fn compress_bound(input_len: usize, prepend_size: Option<bool>) -> usize {
        let n = crate::lz4_impl::block::compress_bound(input_len);
        if prepend_size.unwrap_or(true) {
            n + 4
        } else {
            n
        }
    }

    /// Decompress an LZ4 block whose 4-byte uncompressed length is prepended.
    pub fn decompress_vec(input: &[u8]) -> Result<Vec<u8>, Error> {
        if input.len() < 4 {
            return Err(Error::new(
                std::io::ErrorKind::InvalidInput,
                "Input not long enough",
            ));
        }
        let bytes: [u8; 4] = input[..4].try_into().unwrap();
        let len = u32::from_le_bytes(bytes) as usize;
        let mut buf = Vec::with_capacity(len + crate::lz4_impl::block::OUT_SLACK);
        crate::lz4_impl::block::decompress_block(&input[4..], &mut buf)?;
        if buf.len() != len {
            return Err(Error::new(
                std::io::ErrorKind::InvalidData,
                format!(
                    "lz4 block: prepended length {} doesn't match decoded {}",
                    len,
                    buf.len()
                ),
            ));
        }
        Ok(buf)
    }

    /// Decompress an LZ4 block into a pre-allocated buffer.
    /// `size_prepended == Some(true)` means the input begins with a 4-byte
    /// length header (which we use to allocate but otherwise ignore).
    pub fn decompress_into(
        input: &[u8],
        output: &mut [u8],
        size_prepended: Option<bool>,
    ) -> Result<usize, Error> {
        let block_input = if size_prepended.unwrap_or(false) {
            if input.len() < 4 {
                return Err(Error::new(
                    std::io::ErrorKind::InvalidInput,
                    "Input not long enough for prepended size",
                ));
            }
            // Honour the stored size like `LZ4_decompress_safe` callers do:
            // the caller's buffer must hold at least that much.
            let stored = u32::from_le_bytes([input[0], input[1], input[2], input[3]]) as usize;
            if stored > output.len() {
                return Err(Error::new(
                    std::io::ErrorKind::InvalidData,
                    format!("lz4 block: output buffer too small ({} < stored size {})", output.len(), stored),
                ));
            }
            &input[4..]
        } else {
            input
        };

        // Decompress into a Vec then copy into the slice — the block decoder
        // operates on Vec<u8> for the run-length copy logic.
        let mut tmp: Vec<u8> = Vec::with_capacity(output.len() + crate::lz4_impl::block::OUT_SLACK);
        crate::lz4_impl::block::decompress_block(block_input, &mut tmp)?;
        if tmp.len() > output.len() {
            return Err(Error::new(
                std::io::ErrorKind::InvalidData,
                "lz4 block: output buffer too small",
            ));
        }
        output[..tmp.len()].copy_from_slice(&tmp);
        Ok(tmp.len())
    }

    /// Compress into a fresh Vec.  Optionally prepends the 4-byte
    /// uncompressed length so `decompress_vec` can recover it.
    pub fn compress_vec(
        input: &[u8],
        _level: Option<u32>,
        _acceleration: Option<i32>,
        prepend_size: Option<bool>,
    ) -> Result<Vec<u8>, Error> {
        let prepend_size = prepend_size.unwrap_or(true);
        let mut out = Vec::with_capacity(compress_bound(input.len(), Some(prepend_size)));
        if prepend_size {
            out.extend_from_slice(&(input.len() as u32).to_le_bytes());
        }
        crate::lz4_impl::block::compress_block(input, &mut out);
        Ok(out)
    }

    /// Compress into a pre-allocated buffer.
    pub fn compress_into(
        input: &[u8],
        output: &mut [u8],
        _level: Option<u32>,
        _acceleration: Option<i32>,
        prepend_size: Option<bool>,
    ) -> Result<usize, Error> {
        let prepend_size = prepend_size.unwrap_or(true);
        let need = compress_bound(input.len(), Some(prepend_size));
        if output.len() < need {
            return Err(Error::new(
                std::io::ErrorKind::InvalidInput,
                "lz4 block: output buffer too small",
            ));
        }
        let mut out = Vec::with_capacity(need);
        if prepend_size {
            out.extend_from_slice(&(input.len() as u32).to_le_bytes());
        }
        crate::lz4_impl::block::compress_block(input, &mut out);
        output[..out.len()].copy_from_slice(&out);
        Ok(out.len())
    }

    #[cfg(test)]
    mod tests {
        use super::{compress_vec, decompress_into, decompress_vec};

        const DATA: &[u8; 14] = b"howdy neighbor";

        #[test]
        fn round_trip_store_size() {
            let compressed = compress_vec(DATA, None, None, Some(true)).unwrap();
            let decompressed = decompress_vec(&compressed).unwrap();
            assert_eq!(&decompressed, DATA);
        }
        #[test]
        fn round_trip_no_store_size() {
            let compressed = compress_vec(DATA, None, None, Some(false)).unwrap();
            assert!(decompress_vec(&compressed).is_err());

            let mut decompressed = vec![0u8; DATA.len()];
            decompress_into(&compressed, &mut decompressed, Some(false)).unwrap();
            assert_eq!(&decompressed, DATA);

            let mut decompressed = vec![0u8; DATA.len() + 5_000];
            let n = decompress_into(&compressed, &mut decompressed, Some(false)).unwrap();
            assert_eq!(&decompressed[..n], DATA);
        }
    }
}
