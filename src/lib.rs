#[cfg(any(
    feature = "blosc2",
    feature = "blosc2-static",
    feature = "blosc2-shared"
))]
pub mod blosc2;
#[cfg(feature = "brotli")]
pub mod brotli;
#[cfg(feature = "bzip2")]
pub mod bzip2;
#[cfg(feature = "capi")]
mod capi;

/// Test hook: lz4 frame encoder with explicit frame options.
#[cfg(feature = "lz4")]
#[doc(hidden)]
pub fn lz4_frame_opts_for_tests(input: &[u8], level: Option<u32>, block_linked: bool, content_checksum: bool) -> Vec<u8> {
    lz4_impl::frame::encode_frame_opts(input, level, block_linked, content_checksum)
}

/// Run `f` with a thread-local scratch `Vec<u8>` (cleared, capacity kept)
/// and then write its contents to `output`.
///
/// The generic `Read`/`Write` decompress API forces every call through an
/// intermediate buffer. Allocating that buffer fresh each call costs a page
/// fault (plus kernel zeroing) per 4 KiB of output — a few ms per 10 MB,
/// often 10-20% of a decode. Reusing one per thread keeps the pages mapped
/// across calls, which is what the old C streaming wrappers effectively got
/// from their fixed 128 KiB buffers. Buffers above `SCRATCH_KEEP_MAX` are
/// released after use so a single huge decode doesn't pin memory forever.
#[cfg(any(feature = "lz4", feature = "zstd", feature = "bzip2", feature = "xz", feature = "deflate-static", feature = "deflate-shared"))]
pub(crate) fn with_scratch<W: std::io::Write + ?Sized>(
    output: &mut W,
    f: impl FnOnce(&mut Vec<u8>) -> std::io::Result<()>,
) -> std::io::Result<usize> {
    scratch_with(|buf| {
        f(buf)?;
        output.write_all(buf)?;
        Ok(buf.len())
    })
}

/// Two retained thread-local buffers (input copy + encoded output) for the
/// generic `Read`/`Write` wrappers of the fast codecs: a fresh 50 MB `Vec`
/// per call is ~13k page faults (glibc mmaps/unmaps at that size), which on
/// lz4/zstd is as expensive as the codec itself. Both are cleared on entry.
#[cfg(any(feature = "lz4", feature = "zstd"))]
pub(crate) fn scratch_pair_with<R>(f: impl FnOnce(&mut Vec<u8>, &mut Vec<u8>) -> std::io::Result<R>) -> std::io::Result<R> {
    use std::cell::RefCell;
    const SCRATCH_KEEP_MAX: usize = 64 << 20;
    thread_local! {
        static A: RefCell<Vec<u8>> = const { RefCell::new(Vec::new()) };
        static B: RefCell<Vec<u8>> = const { RefCell::new(Vec::new()) };
    }
    A.with(|ca| {
        B.with(|cb| {
            let mut ga = ca.try_borrow_mut().ok();
            let mut gb = cb.try_borrow_mut().ok();
            let (mut ta, mut tb) = (Vec::new(), Vec::new());
            let a: &mut Vec<u8> = ga.as_deref_mut().unwrap_or(&mut ta);
            let b: &mut Vec<u8> = gb.as_deref_mut().unwrap_or(&mut tb);
            a.clear();
            b.clear();
            let out = f(a, b);
            if a.capacity() > SCRATCH_KEEP_MAX {
                *a = Vec::new();
            }
            if b.capacity() > SCRATCH_KEEP_MAX {
                *b = Vec::new();
            }
            out
        })
    })
}

/// Raw access to the thread-local scratch buffer (cleared on entry).
#[cfg(any(feature = "lz4", feature = "zstd", feature = "bzip2", feature = "xz", feature = "deflate-static", feature = "deflate-shared"))]
pub(crate) fn scratch_with<R>(f: impl FnOnce(&mut Vec<u8>) -> std::io::Result<R>) -> std::io::Result<R> {
    use std::cell::RefCell;
    const SCRATCH_KEEP_MAX: usize = 64 << 20;
    thread_local! {
        static SCRATCH: RefCell<Vec<u8>> = const { RefCell::new(Vec::new()) };
    }
    SCRATCH.with(|cell| {
        // `try_borrow_mut` so a (hypothetical) re-entrant call just gets a
        // fresh temporary instead of panicking.
        let mut guard = cell.try_borrow_mut().ok();
        let mut tmp = Vec::new();
        let buf: &mut Vec<u8> = match guard.as_deref_mut() {
            Some(b) => b,
            None => &mut tmp,
        };
        buf.clear();
        let out = f(buf);
        if buf.capacity() > SCRATCH_KEEP_MAX {
            *buf = Vec::new();
        }
        out
    })
}

/// Streaming-output helper shared by the frame decoders: decoders append to
/// a scratch `Vec`; `maybe_flush` writes everything but the last `keep`
/// bytes (the LZ window) to the sink once more than `chunk` bytes are
/// pending and compacts the buffer. Keeps the decoder's working set at a
/// few MB (cache-resident) instead of the whole output, which otherwise
/// costs ~3x the DRAM traffic on outputs larger than L3 — this is what C's
/// streaming decoders get from their ring buffers.
#[cfg(any(feature = "lz4", feature = "zstd", feature = "bzip2", feature = "xz", feature = "deflate-static", feature = "deflate-shared"))]
pub(crate) struct Streamer<'a> {
    pub sink: Option<&'a mut dyn std::io::Write>,
    keep: usize,
    chunk: usize,
    /// Bytes written to the sink so far.
    pub flushed: usize,
}

#[cfg(any(feature = "lz4", feature = "zstd", feature = "bzip2", feature = "xz", feature = "deflate-static", feature = "deflate-shared"))]
impl<'a> Streamer<'a> {
    pub fn new(sink: Option<&'a mut dyn std::io::Write>, keep: usize, chunk: usize) -> Self {
        Self { sink, keep, chunk, flushed: 0 }
    }

    /// Flush `buf` down to `keep` bytes when more than `keep + chunk` are
    /// pending. `on_flush` sees every flushed byte exactly once (checksums).
    #[inline]
    pub fn maybe_flush(&mut self, buf: &mut Vec<u8>, mut on_flush: impl FnMut(&[u8])) -> std::io::Result<()> {
        if let Some(sink) = self.sink.as_mut() {
            if buf.len() > self.keep + self.chunk {
                let n = buf.len() - self.keep;
                on_flush(&buf[..n]);
                sink.write_all(&buf[..n])?;
                self.flushed += n;
                buf.copy_within(n.., 0);
                buf.truncate(self.keep);
            }
        }
        Ok(())
    }
}

/// Sized adapter so an unsized `&mut W` can be passed as `&mut dyn Write`.
#[cfg(any(feature = "lz4", feature = "zstd", feature = "bzip2", feature = "xz", feature = "deflate-static", feature = "deflate-shared"))]
pub(crate) struct SinkRef<'a, W: std::io::Write + ?Sized>(pub &'a mut W);

#[cfg(any(feature = "lz4", feature = "zstd", feature = "bzip2", feature = "xz", feature = "deflate-static", feature = "deflate-shared"))]
impl<W: std::io::Write + ?Sized> std::io::Write for SinkRef<'_, W> {
    #[inline]
    fn write(&mut self, b: &[u8]) -> std::io::Result<usize> {
        self.0.write(b)
    }
    #[inline]
    fn write_all(&mut self, b: &[u8]) -> std::io::Result<()> {
        self.0.write_all(b)
    }
    #[inline]
    fn flush(&mut self) -> std::io::Result<()> {
        self.0.flush()
    }
}

// Shared runtime CPU feature detection + SIMD wildcopy kernel. No-op on
// non-x86_64/aarch64 targets. Used by the pure-Rust lz4, zstd, xz and deflate
// (inflate) decoders, so it must be compiled whenever any of those is enabled.
#[cfg(any(
    feature = "lz4",
    feature = "zstd",
    feature = "xz",
    feature = "xz-static",
    feature = "xz-shared",
    feature = "deflate",
    feature = "deflate-static",
    feature = "deflate-shared",
    feature = "gzip",
    feature = "gzip-static",
    feature = "gzip-shared",
    feature = "zlib",
    feature = "zlib-static",
    feature = "zlib-shared",
))]
pub(crate) mod cpu_features;

// Pure-Rust DEFLATE / gzip / zlib implementation (no C dependencies).
#[cfg(any(
    feature = "deflate",
    feature = "deflate-static",
    feature = "deflate-shared",
    feature = "gzip",
    feature = "gzip-static",
    feature = "gzip-shared",
    feature = "zlib",
    feature = "zlib-static",
    feature = "zlib-shared",
))]
pub(crate) mod deflate_impl;

// Pure-Rust Zstandard implementation (no C dependencies).
#[cfg(feature = "zstd")]
pub(crate) mod zstd_impl;

// Pure-Rust LZ4 implementation (no C dependencies).
#[cfg(feature = "lz4")]
pub(crate) mod lz4_impl;

// Pure-Rust bzip2 implementation (no C dependencies).
#[cfg(feature = "bzip2")]
pub mod bzip2_impl;

// Pure-Rust XZ / LZMA implementation (no C dependencies).
#[cfg(any(feature = "xz", feature = "xz-static", feature = "xz-shared"))]
pub mod xz_impl;

#[cfg(any(
    feature = "deflate",
    feature = "deflate-static",
    feature = "deflate-shared"
))]
pub mod deflate;
#[cfg(any(feature = "gzip", feature = "gzip-static", feature = "gzip-shared"))]
pub mod gzip;
#[cfg(all(
    any(
        feature = "ideflate",
        feature = "ideflate-static",
        feature = "ideflate-shared"
    ),
    target_pointer_width = "64"
))]
pub mod ideflate;
#[cfg(all(
    any(feature = "igzip", feature = "igzip-static", feature = "igzip-shared"),
    target_pointer_width = "64"
))]
pub mod igzip;
#[cfg(all(
    any(feature = "izlib", feature = "izlib-static", feature = "izlib-shared"),
    target_pointer_width = "64"
))]
pub mod izlib;
#[cfg(feature = "lz4")]
pub mod lz4;
#[cfg(feature = "snappy")]
pub mod snappy;
#[cfg(any(feature = "xz", feature = "xz-static", feature = "xz-shared"))]
pub mod xz;
#[cfg(any(feature = "zlib", feature = "zlib-static", feature = "zlib-shared"))]
pub mod zlib;
#[cfg(feature = "zstd")]
pub mod zstd;

#[cfg(test)]
mod tests {

    use std::io::Cursor;
    use std::str::FromStr;

    // Generate some 'real-world' data by reading src code and duplicating until well over buf size
    static LARGE_DATA: std::sync::LazyLock<Vec<u8>> = std::sync::LazyLock::new(|| {
        // use src code as base, and we have at least 2mb of data
        let mut bytes = read_dir_files(std::path::PathBuf::from_str("./src").unwrap());
        while bytes.len() < 5e6 as usize {
            bytes.extend(bytes.clone());
        }
        bytes
    });

    fn read_dir_files(dir: std::path::PathBuf) -> Vec<u8> {
        let mut all_bytes = vec![];
        for entry in std::fs::read_dir(dir).unwrap().into_iter() {
            let entry = entry.unwrap();
            if entry.file_type().unwrap().is_file() {
                all_bytes.extend(std::fs::read(entry.path()).unwrap());
            } else if entry.file_type().unwrap().is_dir() {
                all_bytes.extend(read_dir_files(entry.path()));
            }
        }
        all_bytes
    }

    // Default testing data
    fn gen_data() -> Vec<u8> {
        (&*LARGE_DATA).clone()
    }

    // Single test generation
    macro_rules! round_trip {
        ($name:ident($compress_output:ident -> $decompress_output:ident), variant=$variant:ident, $(, $args:ident)*) => {
            #[test]
            fn $name() {
                let data = gen_data();

                let mut compressed = Vec::new();

                let compressed_size = if stringify!($decompress_output) == "Slice" {
                        compressed = (0..data.len()).map(|_| 0).collect::<Vec<u8>>();
                        let mut cursor = Cursor::new(compressed.as_mut_slice());
                        crate::$variant::compress(&mut Cursor::new(data.as_slice()), &mut cursor $(, $args)*).unwrap()
                    } else {
                        crate::$variant::compress(&mut Cursor::new(data.as_slice()), &mut Cursor::new(&mut compressed) $(, $args)*).unwrap()
                    };

                println!("Compressed size: {}", compressed_size);
                compressed.truncate(compressed_size);

                let mut decompressed = Vec::new();

                let decompressed_size = if stringify!($decompress_output) == "Slice" {
                        decompressed = (0..data.len()).map(|_| 0).collect::<Vec<u8>>();
                        let mut cursor = Cursor::new(decompressed.as_mut_slice());
                        crate::$variant::decompress(&mut Cursor::new(&compressed), &mut cursor).unwrap()
                    } else {
                        crate::$variant::decompress(&mut Cursor::new(&compressed), &mut decompressed).unwrap()
                    };
                assert_eq!(decompressed_size, data.len());
                if &decompressed[..decompressed_size] != &data {
                    panic!("Decompressed and original data do not match! :-(")
                }
            }
        }
    }

    // macro to generate each variation of Output::* roundtrip.
    macro_rules! test_variant {
        ($variant:ident $(, $args:tt)*) => {
         #[cfg(test)]
         mod $variant {
            use super::*;
            round_trip!(roundtrip_compress_via_slice_decompress_via_slice(Slice -> Slice), variant=$variant, $(, $args)* );
            round_trip!(roundtrip_compress_via_slice_decompress_via_vector(Slice -> Vector), variant=$variant, $(, $args)* );
            round_trip!(roundtrip_compress_via_vector_decompress_via_slice(Vector -> Slice), variant=$variant, $(, $args)* );
            round_trip!(roundtrip_compress_via_vector_decompress_via_vector(Vector -> Vector), variant=$variant, $(, $args)* );
         }
        }
    }

    // Expected compressed_len, subsequent args are supplied to the variant's `compress` call.
    #[cfg(feature = "snappy")]
    test_variant!(snappy);

    #[cfg(feature = "gzip")]
    test_variant!(gzip, None);

    #[cfg(all(
        any(feature = "igzip", feature = "igzip-static", feature = "igzip-shared"),
        target_pointer_width = "64"
    ))]
    test_variant!(igzip, None);

    #[cfg(all(
        any(
            feature = "ideflate",
            feature = "ideflate-static",
            feature = "ideflate-shared"
        ),
        target_pointer_width = "64"
    ))]
    test_variant!(ideflate, None);

    #[cfg(all(
        any(feature = "izlib", feature = "izlib-static", feature = "izlib-shared"),
        target_pointer_width = "64"
    ))]
    test_variant!(izlib, None);

    #[cfg(feature = "brotli")]
    test_variant!(brotli, None);

    #[cfg(feature = "bzip2")]
    test_variant!(bzip2, None);

    #[cfg(feature = "deflate")]
    test_variant!(deflate, None);

    #[cfg(feature = "zstd")]
    test_variant!(zstd, None, None);

    #[cfg(feature = "zlib")]
    test_variant!(zlib, None);

    #[cfg(feature = "lz4")]
    test_variant!(lz4, None);

    #[cfg(feature = "blosc2")]
    test_variant!(blosc2);

    #[cfg(feature = "xz")]
    #[allow(non_upper_case_globals)]
    const format: Option<crate::xz::Format> = None;

    #[allow(non_upper_case_globals)]
    #[cfg(feature = "xz")]
    const check: Option<crate::xz::Check> = None;

    #[allow(non_upper_case_globals)]
    #[cfg(feature = "xz")]
    const filters: Option<crate::xz::Filters> = None;

    #[allow(non_upper_case_globals)]
    #[cfg(feature = "xz")]
    const opts: Option<crate::xz::LzmaOptions> = None;

    #[cfg(feature = "xz")]
    test_variant!(xz, None, format, check, filters, opts);
}

/// Bench/diagnostic helper: single-block LZ4 fast compression (no frame).
#[cfg(feature = "lz4")]
pub fn lz4_impl_block_fast(input: &[u8], output: &mut Vec<u8>) -> usize {
    lz4_impl::block::compress_block(input, output)
}

/// Bench/diagnostic helper: single-block LZ4 HC compression (no frame).
#[cfg(feature = "lz4")]
pub fn lz4_impl_block_hc(input: &[u8], output: &mut Vec<u8>, level: u32) -> usize {
    lz4_impl::block::compress_block_hc(input, output, level)
}
