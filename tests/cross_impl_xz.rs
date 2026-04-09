//! Cross-implementation tests for the pure-Rust XZ / LZMA codec.
//!
//! These compress with C `xz2` (still kept as a dev-dependency) and
//! decompress with our pure-Rust `xz_impl`.  Once the encoder lands we'll
//! also add ours→C round-trips here, mirroring the bzip2 test layout.

use std::io::Write;

fn gen_text(size: usize) -> Vec<u8> {
    let phrases: &[&[u8]] = &[
        b"The quick brown fox jumps over the lazy dog. ",
        b"Lorem ipsum dolor sit amet, consectetur adipiscing elit. ",
        b"fn main() { println!(\"hello world\"); }\n",
    ];
    let mut data = Vec::with_capacity(size);
    for phrase in phrases.iter().cycle() {
        let take = phrase.len().min(size - data.len());
        data.extend_from_slice(&phrase[..take]);
        if data.len() >= size {
            break;
        }
    }
    data
}

fn c_xz_compress(data: &[u8], level: u32) -> Vec<u8> {
    let mut enc = xz2::write::XzEncoder::new(Vec::new(), level);
    enc.write_all(data).unwrap();
    enc.finish().unwrap()
}

fn our_decompress(data: &[u8]) -> Vec<u8> {
    libcramjam::xz_impl::decode_xz(data).unwrap()
}

fn our_compress(data: &[u8], preset: u32) -> Vec<u8> {
    libcramjam::xz_impl::encode_xz(data, preset).unwrap()
}

fn c_xz_decompress(data: &[u8]) -> Vec<u8> {
    use std::io::Read;
    let mut dec = xz2::read::XzDecoder::new(data);
    let mut out = Vec::new();
    dec.read_to_end(&mut out).unwrap();
    out
}

#[test]
fn c_compress_our_decompress_tiny() {
    let data = b"abc".to_vec();
    let compressed = c_xz_compress(&data, 6);
    let decompressed = our_decompress(&compressed);
    assert_eq!(decompressed, data);
}

#[test]
fn c_compress_our_decompress_hello() {
    let data = b"hello world".to_vec();
    let compressed = c_xz_compress(&data, 6);
    let decompressed = our_decompress(&compressed);
    assert_eq!(decompressed, data);
}

#[test]
fn c_compress_our_decompress_text_small() {
    for level in [1u32, 6, 9] {
        let data = gen_text(1_000);
        let compressed = c_xz_compress(&data, level);
        let decompressed = our_decompress(&compressed);
        assert_eq!(decompressed, data, "level={}", level);
    }
}

#[test]
fn c_compress_our_decompress_text_medium() {
    for level in [1u32, 6, 9] {
        let data = gen_text(100_000);
        let compressed = c_xz_compress(&data, level);
        let decompressed = our_decompress(&compressed);
        assert_eq!(decompressed, data, "level={}", level);
    }
}

#[test]
fn c_compress_our_decompress_repeated() {
    let data = vec![0xAAu8; 5000];
    let compressed = c_xz_compress(&data, 6);
    let decompressed = our_decompress(&compressed);
    assert_eq!(decompressed, data);
}

#[test]
fn c_compress_our_decompress_random_small() {
    let mut s: u32 = 0xCAFE_BABE;
    let data: Vec<u8> = (0..2000)
        .map(|_| {
            s ^= s << 13;
            s ^= s >> 17;
            s ^= s << 5;
            (s >> 16) as u8
        })
        .collect();
    let compressed = c_xz_compress(&data, 6);
    let decompressed = our_decompress(&compressed);
    assert_eq!(decompressed, data);
}

#[test]
fn c_compress_our_decompress_src_dir() {
    fn read_dir_files(dir: std::path::PathBuf) -> Vec<u8> {
        let mut all = Vec::new();
        for entry in std::fs::read_dir(dir).unwrap() {
            let entry = entry.unwrap();
            if entry.file_type().unwrap().is_file() {
                all.extend(std::fs::read(entry.path()).unwrap());
            } else if entry.file_type().unwrap().is_dir() {
                all.extend(read_dir_files(entry.path()));
            }
        }
        all
    }
    let bytes = read_dir_files(std::path::PathBuf::from("./src"));
    for level in [1u32, 5, 9] {
        let compressed = c_xz_compress(&bytes, level);
        let decompressed = our_decompress(&compressed);
        assert_eq!(decompressed, bytes, "src dir failed: level={}", level);
    }
}

#[test]
fn c_compress_our_decompress_text_large() {
    // 1 MiB of phrase-cycled text — triggers multi-block at preset 1.
    let data = gen_text(1_000_000);
    for level in [1u32, 6, 9] {
        let compressed = c_xz_compress(&data, level);
        let decompressed = our_decompress(&compressed);
        assert_eq!(decompressed, data, "level={}", level);
    }
}

#[test]
fn c_compress_our_decompress_random_large() {
    let mut s: u32 = 0xDEAD_BEEF;
    let data: Vec<u8> = (0..200_000)
        .map(|_| {
            s ^= s << 13;
            s ^= s >> 17;
            s ^= s << 5;
            (s >> 16) as u8
        })
        .collect();
    let compressed = c_xz_compress(&data, 6);
    let decompressed = our_decompress(&compressed);
    assert_eq!(decompressed, data);
}

#[test]
fn c_compress_our_decompress_empty() {
    let data: Vec<u8> = vec![];
    let compressed = c_xz_compress(&data, 6);
    let decompressed = our_decompress(&compressed);
    assert_eq!(decompressed, data);
}

// =========================================================================
// Legacy LZMA "Alone" (.lzma) format
// =========================================================================
//
// The xz2 crate exposes the alone format via Stream::new_lzma_encoder /
// LzmaOptions::new_preset.  Our decoder is supposed to auto-detect ALONE
// vs XZ from the leading bytes and route appropriately.

fn c_alone_compress(data: &[u8], preset: u32) -> Vec<u8> {
    use std::io::Write;
    let opts = xz2::stream::LzmaOptions::new_preset(preset).unwrap();
    let stream = xz2::stream::Stream::new_lzma_encoder(&opts).unwrap();
    let mut enc = xz2::write::XzEncoder::new_stream(Vec::new(), stream);
    enc.write_all(data).unwrap();
    enc.finish().unwrap()
}

#[test]
fn c_alone_compress_our_decompress_tiny() {
    let data = b"hello world".to_vec();
    let compressed = c_alone_compress(&data, 6);
    // First byte should NOT be the xz magic 0xFD.
    assert_ne!(compressed[0], 0xFD, "expected ALONE format, not xz");
    let decompressed = our_decompress(&compressed);
    assert_eq!(decompressed, data);
}

/// Regression: the 2-byte input `b"x\0"` compressed in ALONE format
/// crashed our decoder with "bad stream header magic" because the
/// stream had no end-of-payload marker AND no known size beyond what
/// the alone header declares.
#[test]
fn c_alone_compress_our_decompress_x_null() {
    let data = b"x\0".to_vec();
    let compressed = c_alone_compress(&data, 6);
    eprintln!("[x_null] compressed = {:02x?}", compressed);
    let decompressed = our_decompress(&compressed);
    assert_eq!(decompressed, data);
}

#[test]
fn c_alone_compress_our_decompress_text() {
    for level in [1u32, 6, 9] {
        let data = gen_text(100_000);
        let compressed = c_alone_compress(&data, level);
        assert_ne!(compressed[0], 0xFD);
        let decompressed = our_decompress(&compressed);
        assert_eq!(decompressed, data, "level={}", level);
    }
}

#[test]
fn c_xz_compress_our_decompress_mozilla() {
    let raw = match std::fs::read("/tmp/mozilla.raw") {
        Ok(r) => r,
        Err(_) => {
            eprintln!("[skip] /tmp/mozilla.raw missing");
            return;
        }
    };
    eprintln!("[mozilla xz] input: {} MB", raw.len() / 1024 / 1024);
    let compressed = c_xz_compress(&raw, 6);
    eprintln!("[mozilla xz] compressed: {} MB", compressed.len() / 1024 / 1024);
    let decompressed = our_decompress(&compressed);
    assert_eq!(decompressed.len(), raw.len(), "length mismatch");
    if decompressed != raw {
        let first = decompressed.iter().zip(raw.iter()).position(|(a, b)| a != b).unwrap();
        panic!("byte mismatch at offset {}", first);
    }
}

/// Find where our decoder diverges from xz2 on the mozilla benchmark.
/// Inspects partial output by reaching into the LZMA2 driver directly.
#[test]
fn mozilla_decoder_divergence() {
    let raw = match std::fs::read("/tmp/mozilla.raw") {
        Ok(r) => r,
        Err(_) => return,
    };
    let sub = &raw[..460 * 1024];
    let compressed = c_xz_compress(sub, 6);
    eprintln!("[divergence] input={} compressed={}", sub.len(), compressed.len());

    let c_decoded = c_xz_decompress(&compressed);
    assert_eq!(c_decoded, sub, "xz2 decoded mismatch");

    // Decode-with-partial-capture: try our decoder and capture whatever
    // it manages to produce before erroring.
    let mut our_partial = Vec::new();
    let _ = libcramjam::xz_impl::xz_format::decode_xz_stream(&compressed, &mut our_partial);
    eprintln!("[divergence] ours produced {} bytes before error", our_partial.len());

    // Find first divergence.
    let n = our_partial.len().min(c_decoded.len());
    let first_diff = (0..n).find(|&i| our_partial[i] != c_decoded[i]);
    match first_diff {
        Some(idx) => {
            eprintln!("[divergence] first mismatch at offset {} (chunk-relative {})",
                idx, idx as i64 - 402409);
            let lo = idx.saturating_sub(16);
            let hi = (idx + 16).min(n);
            eprintln!("  ours[{}..{}]:   {:02x?}", lo, hi, &our_partial[lo..hi]);
            eprintln!("  xz2 [{}..{}]:   {:02x?}", lo, hi, &c_decoded[lo..hi]);
        }
        None => {
            eprintln!("[divergence] all {} bytes match before our decoder errored", n);
            if our_partial.len() < c_decoded.len() {
                eprintln!("  next expected byte = 0x{:02x}", c_decoded[n]);
            }
        }
    }
}

#[test]
fn c_alone_compress_our_decompress_random() {
    let mut s: u32 = 0xCAFE_BABE;
    let data: Vec<u8> = (0..50_000)
        .map(|_| {
            s ^= s << 13;
            s ^= s >> 17;
            s ^= s << 5;
            (s >> 16) as u8
        })
        .collect();
    let compressed = c_alone_compress(&data, 6);
    assert_ne!(compressed[0], 0xFD);
    let decompressed = our_decompress(&compressed);
    assert_eq!(decompressed, data);
}

#[test]
fn c_compress_our_decompress_single_byte() {
    let data = vec![0x42u8];
    let compressed = c_xz_compress(&data, 6);
    let decompressed = our_decompress(&compressed);
    assert_eq!(decompressed, data);
}

// =========================================================================
// Ours → ours round-trip
// =========================================================================

#[test]
fn ours_roundtrip_tiny() {
    let data = b"abc".to_vec();
    let compressed = our_compress(&data, 6);
    let decompressed = our_decompress(&compressed);
    assert_eq!(decompressed, data);
}

#[test]
fn ours_roundtrip_hello() {
    let data = b"hello world hello world hello world".to_vec();
    let compressed = our_compress(&data, 6);
    let decompressed = our_decompress(&compressed);
    assert_eq!(decompressed, data);
}

#[test]
fn ours_roundtrip_text_small() {
    for level in [1u32, 6, 9] {
        let data = gen_text(1_000);
        let compressed = our_compress(&data, level);
        let decompressed = our_decompress(&compressed);
        assert_eq!(decompressed, data, "level={}", level);
    }
}

#[test]
fn ours_roundtrip_text_medium() {
    for level in [1u32, 6, 9] {
        let data = gen_text(100_000);
        let compressed = our_compress(&data, level);
        let decompressed = our_decompress(&compressed);
        assert_eq!(decompressed, data, "level={}", level);
    }
}

#[test]
fn ours_roundtrip_repeated() {
    let data = vec![0xAAu8; 5000];
    let compressed = our_compress(&data, 6);
    let decompressed = our_decompress(&compressed);
    assert_eq!(decompressed, data);
}

#[test]
fn ours_roundtrip_random_small() {
    let mut s: u32 = 0xCAFE_BABE;
    let data: Vec<u8> = (0..2000)
        .map(|_| {
            s ^= s << 13;
            s ^= s >> 17;
            s ^= s << 5;
            (s >> 16) as u8
        })
        .collect();
    let compressed = our_compress(&data, 6);
    let decompressed = our_decompress(&compressed);
    assert_eq!(decompressed, data);
}

#[test]
fn ours_roundtrip_empty() {
    let data: Vec<u8> = vec![];
    let compressed = our_compress(&data, 6);
    let decompressed = our_decompress(&compressed);
    assert_eq!(decompressed, data);
}

// =========================================================================
// Ours → C round-trip (the real cross-impl proof)
// =========================================================================

#[test]
fn our_compress_c_decompress_tiny() {
    let data = b"abc".to_vec();
    let compressed = our_compress(&data, 6);
    let decompressed = c_xz_decompress(&compressed);
    assert_eq!(decompressed, data);
}

#[test]
fn our_compress_c_decompress_text_small() {
    for level in [1u32, 6, 9] {
        let data = gen_text(1_000);
        let compressed = our_compress(&data, level);
        let decompressed = c_xz_decompress(&compressed);
        assert_eq!(decompressed, data, "level={}", level);
    }
}

#[test]
fn our_compress_c_decompress_text_medium() {
    for level in [1u32, 6, 9] {
        let data = gen_text(100_000);
        let compressed = our_compress(&data, level);
        let decompressed = c_xz_decompress(&compressed);
        assert_eq!(decompressed, data, "level={}", level);
    }
}

/// Inspect what kind of LZMA2 chunks C xz produces for various inputs.
/// Used to verify our optimization assumptions (does C use uncompressed
/// chunks for incompressible data? what mode flags does it pick?).
#[test]
fn inspect_c_chunks() {
    fn dump(name: &str, data: &[u8], level: u32) {
        let compressed = c_xz_compress(data, level);
        let block_header_size = (compressed[12] as usize + 1) * 4;
        let lzma2_start = 12 + block_header_size;
        let mut pos = lzma2_start;
        let mut chunk_idx = 0;
        eprintln!("[chunks] {} L{} compressed_len={} lzma2_start={}", name, level, compressed.len(), lzma2_start);
        while pos < compressed.len() {
            let control = compressed[pos];
            if control == 0x00 {
                eprintln!("  chunk {}: END", chunk_idx);
                break;
            }
            if control == 0x01 || control == 0x02 {
                let size = ((compressed[pos+1] as usize) << 8 | compressed[pos+2] as usize) + 1;
                eprintln!("  chunk {}: UNCOMPRESSED ({}), size={}",
                    chunk_idx, if control == 0x01 { "dict reset" } else { "no reset" }, size);
                pos += 3 + size;
            } else if control >= 0x80 {
                let mode = (control >> 5) & 0x03;
                let unc_high = (control & 0x1F) as u32;
                let unc_size = ((unc_high << 16)
                    | ((compressed[pos+1] as u32) << 8)
                    | compressed[pos+2] as u32) + 1;
                let comp_size = (((compressed[pos+3] as u32) << 8)
                    | compressed[pos+4] as u32) + 1;
                eprintln!("  chunk {}: LZMA mode={} unc={} comp={} ratio={:.2}",
                    chunk_idx, mode, unc_size, comp_size, comp_size as f64 / unc_size as f64);
                let header_size = if mode >= 2 { 6 } else { 5 };
                pos += header_size + comp_size as usize;
            } else {
                eprintln!("  chunk {}: invalid 0x{:02x}", chunk_idx, control);
                break;
            }
            chunk_idx += 1;
        }
    }
    let mut s: u32 = 0xDEAD_BEEF;
    let random: Vec<u8> = (0..100_000)
        .map(|_| { s ^= s << 13; s ^= s >> 17; s ^= s << 5; (s >> 16) as u8 })
        .collect();
    dump("text_100k", &gen_text(100_000), 6);
    dump("random_100k", &random, 6);
}

/// Print compression ratio so we can eyeball the encoder quality vs C xz
/// during development.  Not a hard assertion — just a "did we beat 1.5x
/// of C xz?" sanity check that catches catastrophic regressions.
#[test]
fn ratio_check() {
    fn report(name: &str, data: &[u8], levels: &[u32]) {
        for &level in levels {
            let ours = our_compress(data, level);
            let theirs = c_xz_compress(data, level);
            eprintln!(
                "[xz ratio] {} L{}: ours={} bytes ({:.2}%) c_xz={} bytes ({:.2}%) ours/c_xz={:.2}x",
                name,
                level,
                ours.len(),
                ours.len() as f64 / data.len() as f64 * 100.0,
                theirs.len(),
                theirs.len() as f64 / data.len() as f64 * 100.0,
                ours.len() as f64 / theirs.len() as f64,
            );
        }
    }

    report("text_100k", &gen_text(100_000), &[1, 6, 9]);

    let mut s: u32 = 0xCAFE_BABE;
    let random: Vec<u8> = (0..100_000)
        .map(|_| {
            s ^= s << 13;
            s ^= s >> 17;
            s ^= s << 5;
            (s >> 16) as u8
        })
        .collect();
    report("random_100k", &random, &[1, 6]);

    fn read_dir_files(dir: std::path::PathBuf) -> Vec<u8> {
        let mut all = Vec::new();
        for entry in std::fs::read_dir(dir).unwrap() {
            let entry = entry.unwrap();
            if entry.file_type().unwrap().is_file() {
                all.extend(std::fs::read(entry.path()).unwrap());
            } else if entry.file_type().unwrap().is_dir() {
                all.extend(read_dir_files(entry.path()));
            }
        }
        all
    }
    let src = read_dir_files(std::path::PathBuf::from("./src"));
    report("src_dir", &src, &[1, 6]);
}
