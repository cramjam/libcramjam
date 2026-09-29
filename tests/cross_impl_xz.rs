//! Cross-implementation tests for the pure-Rust XZ / LZMA codec.
//!
//! These compress with C `xz2` (still kept as a dev-dependency) and
//! decompress with our pure-Rust `xz_impl`.  Once the encoder lands we'll
//! also add ours→C round-trips here, mirroring the bzip2 test layout.

// Exercises the pure-Rust internals directly; backend-neutral checks are in backend_contract.rs.
#![cfg(feature = "xz-pure")]

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

// =========================================================================
// BCJ filter cross-impl tests
//
// Each test compresses a synthetic "executable-ish" buffer with C xz2 using
// a custom filter chain (BCJ + LZMA2) and verifies our pure-Rust decoder
// recovers the bytes.  The buffer mixes random bytes with embedded
// branch-target patterns so the BCJ encoder actually has something to
// transform.
// =========================================================================

fn gen_bcj_corpus(seed: u32, size: usize) -> Vec<u8> {
    let mut s = seed;
    let mut out = Vec::with_capacity(size);
    while out.len() < size {
        s ^= s << 13;
        s ^= s >> 17;
        s ^= s << 5;
        // Sprinkle x86-style CALL/JMP opcodes (E8/E9) plus some ARM-BL
        // (last byte 0xEB), PowerPC branch (top 6 bits = 0x12), SPARC
        // CALL (top byte 0x40), and a benign IA-64 bundle every so often
        // — none of which need to encode meaningfully, just to give the
        // BCJ filters something to chew on.
        let r = (s >> 16) as u8;
        match r & 0x07 {
            0 => {
                out.extend_from_slice(&[0xE8, 0x12, 0x34, 0x56, 0x00]);
            }
            1 => {
                out.extend_from_slice(&[0xE9, 0x78, 0x9A, 0xBC, 0xFF]);
            }
            2 => {
                // ARM BL (4 bytes ending in 0xEB).
                out.extend_from_slice(&[0x10, 0x20, 0x30, 0xEB]);
            }
            3 => {
                // PowerPC bl (top byte = 0x48 → 0x12 << 2, low byte LK=1).
                out.extend_from_slice(&[0x48, 0x00, 0x10, 0x01]);
            }
            4 => {
                // SPARC CALL.
                out.extend_from_slice(&[0x40, 0x00, 0x00, 0x10]);
            }
            5 => {
                // IA-64 instruction bundle (16 bytes).
                out.extend_from_slice(&[
                    0x12, 0x00, 0x00, 0x00, 0x05, 0x00, 0x00, 0x00,
                    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
                ]);
            }
            _ => {
                out.push(r);
            }
        }
    }
    out.truncate(size);
    out
}

fn c_xz_compress_with_filter<F>(data: &[u8], build: F) -> Vec<u8>
where
    F: FnOnce(&mut xz2::stream::Filters),
{
    use xz2::stream::{Check, Filters, LzmaOptions, Stream};
    let mut filters = Filters::new();
    build(&mut filters);
    // Always end with LZMA2 (the dev-dep API requires it).
    let opts = LzmaOptions::new_preset(6).unwrap();
    filters.lzma2(&opts);
    let stream = Stream::new_stream_encoder(&filters, Check::Crc64).unwrap();
    let mut enc = xz2::write::XzEncoder::new_stream(Vec::new(), stream);
    enc.write_all(data).unwrap();
    enc.finish().unwrap()
}

fn run_bcj_test(name: &str, build: fn(&mut xz2::stream::Filters)) {
    for size in [256usize, 4096, 64 * 1024] {
        let data = gen_bcj_corpus(0xC0DE_F00D, size);
        let compressed = c_xz_compress_with_filter(&data, build);
        let decompressed = our_decompress(&compressed);
        assert_eq!(
            decompressed, data,
            "BCJ filter {name} failed at size {size}: produced {} bytes",
            decompressed.len()
        );
    }
}

#[test]
fn bcj_x86_cross_impl() {
    run_bcj_test("x86", |f| {
        f.x86();
    });
}

#[test]
fn bcj_arm_cross_impl() {
    run_bcj_test("arm", |f| {
        f.arm();
    });
}

#[test]
fn bcj_arm_thumb_cross_impl() {
    run_bcj_test("arm_thumb", |f| {
        f.arm_thumb();
    });
}

#[test]
fn bcj_powerpc_cross_impl() {
    run_bcj_test("powerpc", |f| {
        f.powerpc();
    });
}

#[test]
fn bcj_sparc_cross_impl() {
    run_bcj_test("sparc", |f| {
        f.sparc();
    });
}

#[test]
fn bcj_ia64_cross_impl() {
    run_bcj_test("ia64", |f| {
        f.ia64();
    });
}

/// `.lzma` (alone) encoder output must decode with liblzma and with ours.
#[test]
fn alone_encoder_roundtrips_through_liblzma() {
    use std::io::{Cursor, Read};
    let mut data: Vec<u8> = b"lzma alone format, the way Python's lzma.compress(format=FORMAT_ALONE) writes it. ".repeat(500);
    data.extend((0..50_000u32).map(|i| ((i.wrapping_mul(2654435761u32)) >> 24) as u8));
    for preset in [0u32, 1, 6, 9] {
        let mut ours = Vec::new();
        libcramjam::xz::compress(
            &mut Cursor::new(&data), &mut ours, Some(preset),
            Some(libcramjam::xz::Format::ALONE), None::<libcramjam::xz::Check>,
            None::<libcramjam::xz::Filters>, None::<libcramjam::xz::LzmaOptions>,
        ).unwrap();
        assert_eq!(ours[0] as u32, (2 * 5 + 0) * 9 + 3, "props byte lc=3 lp=0 pb=2");
        let stream = xz2::stream::Stream::new_lzma_decoder(u64::MAX).unwrap();
        let mut dec = xz2::read::XzDecoder::new_stream(&ours[..], stream);
        let mut out = Vec::new();
        dec.read_to_end(&mut out).unwrap();
        assert_eq!(out, data, "preset {preset}");
        let mut back = Vec::new();
        libcramjam::xz::decompress(&mut Cursor::new(&ours), &mut back).unwrap();
        assert_eq!(back, data);
    }
    // Empty input.
    let mut ours = Vec::new();
    libcramjam::xz::compress(
        &mut Cursor::new(&[][..]), &mut ours, Some(6),
        Some(libcramjam::xz::Format::ALONE), None::<libcramjam::xz::Check>,
        None::<libcramjam::xz::Filters>, None::<libcramjam::xz::LzmaOptions>,
    ).unwrap();
    let stream = xz2::stream::Stream::new_lzma_decoder(u64::MAX).unwrap();
    let mut out = Vec::new();
    xz2::read::XzDecoder::new_stream(&ours[..], stream).read_to_end(&mut out).unwrap();
    assert!(out.is_empty());
}

// ---------------------------------------------------------------------------
// RAW format, BCJ encoders, LZMA1 chains, SHA-256 — ours <-> liblzma
// ---------------------------------------------------------------------------

use std::io::{Cursor, Read};
use libcramjam::xz::{Check, Filters, Format, LzmaOptions};

fn ours_encode(data: &[u8], preset: u32, format: Format, check: Check, filters: Option<Filters>) -> Vec<u8> {
    let mut out = Vec::new();
    libcramjam::xz::compress(
        &mut Cursor::new(data), &mut out, Some(preset), Some(format), Some(check), filters,
        None::<LzmaOptions>,
    ).unwrap();
    out
}

fn ours_decode(data: &[u8]) -> Vec<u8> {
    let mut out = Vec::new();
    libcramjam::xz::decompress(&mut Cursor::new(data), &mut out).unwrap();
    out
}

fn ours_decode_raw(data: &[u8], filters: Filters) -> Vec<u8> {
    let mut out = Vec::new();
    libcramjam::xz::decompress_raw(&mut Cursor::new(data), &mut out, filters).unwrap();
    out
}

fn c_decode_stream(data: &[u8], stream: xz2::stream::Stream) -> Vec<u8> {
    let mut out = Vec::new();
    xz2::read::XzDecoder::new_stream(data, stream).read_to_end(&mut out).unwrap();
    out
}

fn c_encode_stream(data: &[u8], stream: xz2::stream::Stream) -> Vec<u8> {
    let mut enc = xz2::write::XzEncoder::new_stream(Vec::new(), stream);
    enc.write_all(data).unwrap();
    enc.finish().unwrap()
}

/// liblzma's raw coder (`lzma_raw_buffer_encode/decode`; the `xz2` crate
/// doesn't wrap it): `bcj` filter ids, then LZMA1 or LZMA2 at `preset` with
/// optional lc/lp/pb overrides.  `out_cap` bounds the output buffer.
fn c_raw(
    data: &[u8],
    bcj: &[u64],
    lzma1: bool,
    preset: u32,
    lclppb: Option<(u32, u32, u32)>,
    encode: bool,
    out_cap: usize,
) -> Vec<u8> {
    use std::ptr::null_mut;
    unsafe {
        let mut opts: lzma_sys::lzma_options_lzma = std::mem::zeroed();
        assert_eq!(lzma_sys::lzma_lzma_preset(&mut opts, preset), 0);
        if let Some((lc, lp, pb)) = lclppb {
            opts.lc = lc;
            opts.lp = lp;
            opts.pb = pb;
        }
        let mut filters: Vec<lzma_sys::lzma_filter> =
            bcj.iter().map(|&id| lzma_sys::lzma_filter { id, options: null_mut() }).collect();
        filters.push(lzma_sys::lzma_filter {
            id: if lzma1 { lzma_sys::LZMA_FILTER_LZMA1 } else { lzma_sys::LZMA_FILTER_LZMA2 },
            options: &mut opts as *mut _ as *mut std::ffi::c_void,
        });
        filters.push(lzma_sys::lzma_filter { id: lzma_sys::LZMA_VLI_UNKNOWN, options: null_mut() });
        let mut out = vec![0u8; out_cap];
        let mut out_pos = 0usize;
        let ret = if encode {
            lzma_sys::lzma_raw_buffer_encode(
                filters.as_ptr(), std::ptr::null(), data.as_ptr(), data.len(),
                out.as_mut_ptr(), &mut out_pos, out.len(),
            )
        } else {
            let mut in_pos = 0usize;
            let ret = lzma_sys::lzma_raw_buffer_decode(
                filters.as_ptr(), std::ptr::null(), data.as_ptr(), &mut in_pos, data.len(),
                out.as_mut_ptr(), &mut out_pos, out.len(),
            );
            assert_eq!(in_pos, data.len(), "liblzma did not consume the whole raw stream");
            ret
        };
        assert_eq!(ret, lzma_sys::LZMA_OK, "liblzma raw {} failed", if encode { "encode" } else { "decode" });
        out.truncate(out_pos);
        out
    }
}

fn c_raw_encode(data: &[u8], bcj: &[u64], lzma1: bool, preset: u32) -> Vec<u8> {
    c_raw(data, bcj, lzma1, preset, None, true, data.len() * 2 + 4096)
}

fn c_raw_decode(data: &[u8], bcj: &[u64], lzma1: bool, preset: u32, lclppb: Option<(u32, u32, u32)>, len: usize) -> Vec<u8> {
    c_raw(data, bcj, lzma1, preset, lclppb, false, len + 64)
}

/// The six BCJ filters as (name, our chain builder, liblzma filter id).
type OurBuild = fn(&mut Filters) -> &mut Filters;
const BCJ: &[(&str, OurBuild, u64)] = &[
    ("x86", |f| f.x86(), lzma_sys::LZMA_FILTER_X86),
    ("arm", |f| f.arm(), lzma_sys::LZMA_FILTER_ARM),
    ("armthumb", |f| f.arm_thumb(), lzma_sys::LZMA_FILTER_ARMTHUMB),
    ("powerpc", |f| f.powerpc(), lzma_sys::LZMA_FILTER_POWERPC),
    ("sparc", |f| f.sparc(), lzma_sys::LZMA_FILTER_SPARC),
    ("ia64", |f| f.ia64(), lzma_sys::LZMA_FILTER_IA64),
];

/// x86 code-like buffer: CALL/JMP rel32 instructions from sequential
/// positions to a handful of fixed targets, so every relative displacement
/// differs but the BCJ-converted absolute ones repeat.
fn gen_x86_calls(n: usize) -> Vec<u8> {
    let targets = [0x1000u32, 0x2340, 0x8000, 0xABCD0, 0x12345, 0x40000, 0x77777, 0x99990];
    let mut out = Vec::with_capacity(n * 12);
    let mut s = 0x9E37_79B9u32;
    for i in 0..n {
        s ^= s << 13;
        s ^= s >> 17;
        s ^= s << 5;
        // Some filler "instructions" between calls.
        let filler = (s % 7) as usize;
        for j in 0..filler {
            out.push(((s >> (j * 4)) & 0x7F) as u8 | 0x40);
        }
        let pos = out.len() as u32;
        let target = targets[i % targets.len()];
        let rel = target.wrapping_sub(pos + 5);
        out.push(if s & 0x100 != 0 { 0xE8 } else { 0xE9 });
        out.extend_from_slice(&rel.to_le_bytes());
    }
    out
}

#[test]
fn raw_lzma2_cross_impl() {
    let data = gen_text(200_000);
    for preset in [0u32, 1, 6, 9] {
        let opts = LzmaOptions::new_preset(preset).unwrap();
        let raw = ours_encode(&data, preset, Format::RAW, Check::None, None);
        // Decodes with liblzma's raw decoder given the same chain.
        assert_eq!(c_raw_decode(&raw, &[], false, preset, None, data.len()), data, "preset {preset}");
        // And with ours.
        let mut f = Filters::new();
        f.lzma2(&opts);
        assert_eq!(ours_decode_raw(&raw, f), data);
        // liblzma raw -> ours.
        let theirs = c_raw_encode(&data, &[], false, preset);
        let mut f = Filters::new();
        f.lzma2(&opts);
        assert_eq!(ours_decode_raw(&theirs, f), data);
    }
}

#[test]
fn raw_lzma1_cross_impl() {
    let mut data = gen_text(150_000);
    data.extend((0..30_000u32).map(|i| ((i.wrapping_mul(2654435761u32)) >> 24) as u8));
    for preset in [0u32, 3, 6, 9] {
        let opts = LzmaOptions::new_preset(preset).unwrap();
        let mut f = Filters::new();
        f.lzma1(&opts);
        let raw = ours_encode(&data, preset, Format::RAW, Check::None, Some(f.clone()));
        assert_eq!(c_raw_decode(&raw, &[], true, preset, None, data.len()), data, "preset {preset}");
        assert_eq!(ours_decode_raw(&raw, f.clone()), data);
        let theirs = c_raw_encode(&data, &[], true, preset);
        assert_eq!(ours_decode_raw(&theirs, f), data);
    }
    // Non-default lc/lp/pb travel through the chain options.
    let mut opts = LzmaOptions::new_preset(6).unwrap();
    opts.literal_context_bits(0).literal_position_bits(2).position_bits(0);
    let mut f = Filters::new();
    f.lzma1(&opts);
    let raw = ours_encode(&data, 6, Format::RAW, Check::None, Some(f.clone()));
    assert_eq!(c_raw_decode(&raw, &[], true, 6, Some((0, 2, 0)), data.len()), data);
    // With the default lc/lp/pb liblzma either errors or yields garbage.
    let wrong = std::panic::catch_unwind(|| c_raw_decode(&raw, &[], true, 6, None, data.len()));
    assert!(wrong.map(|v| v != data).unwrap_or(true), "default lc/lp/pb must not decode it");
    assert_eq!(ours_decode_raw(&raw, f), data);
}

#[test]
fn bcj_encoders_cross_impl_xz_and_raw() {
    let opts = LzmaOptions::new_preset(6).unwrap();
    for &(name, our_build, c_id) in BCJ {
        for size in [0usize, 5, 4096, 200_000] {
            let data = gen_bcj_corpus(0xBEEF_0000 + size as u32, size);

            // .xz container: BCJ + LZMA2, CRC64.
            let mut f = Filters::new();
            our_build(&mut f).lzma2(&opts);
            let xz = ours_encode(&data, 6, Format::XZ, Check::Crc64, Some(f));
            assert_eq!(c_xz_decompress(&xz), data, "{name} .xz size {size} (liblzma decode)");
            assert_eq!(ours_decode(&xz), data, "{name} .xz size {size} (our decode)");
            // The block header must actually declare the filter.
            let block_flags = xz[13];
            assert_eq!(block_flags & 3, 1, "{name}: block header should list 2 filters");

            // RAW: BCJ + LZMA2 and BCJ + LZMA1, both directions.
            for lzma1 in [false, true] {
                let mut f = Filters::new();
                our_build(&mut f);
                if lzma1 {
                    f.lzma1(&opts);
                } else {
                    f.lzma2(&opts);
                }
                let raw = ours_encode(&data, 6, Format::RAW, Check::None, Some(f.clone()));
                assert_eq!(
                    c_raw_decode(&raw, &[c_id], lzma1, 6, None, data.len()),
                    data,
                    "{name} raw lzma1={lzma1} size {size} (liblzma decode)"
                );
                assert_eq!(ours_decode_raw(&raw, f.clone()), data, "{name} raw lzma1={lzma1} size {size}");
                let theirs = c_raw_encode(&data, &[c_id], lzma1, 6);
                assert_eq!(ours_decode_raw(&theirs, f), data, "{name} raw lzma1={lzma1} size {size} (C encode)");
                // The filtered bytes are identical to liblzma's: peel only
                // the LZMA layer off both raw streams and compare.
                let ours_t = c_raw_decode(&raw, &[], lzma1, 6, None, data.len());
                let theirs_t = c_raw_decode(&theirs, &[], lzma1, 6, None, data.len());
                assert!(ours_t == theirs_t, "{name} transform differs from liblzma (size {size})");
            }
        }
    }
}

#[test]
fn x86_bcj_on_real_call_patterns_matches_liblzma() {
    let data = gen_x86_calls(20_000);
    let opts = LzmaOptions::new_preset(6).unwrap();
    let mut f = Filters::new();
    f.x86().lzma2(&opts);
    let with = ours_encode(&data, 6, Format::XZ, Check::Crc64, Some(f));
    let without = ours_encode(&data, 6, Format::XZ, Check::Crc64, None);
    assert_eq!(c_xz_decompress(&with), data);
    assert_eq!(ours_decode(&with), data);
    assert!(with.len() < without.len() * 3 / 4, "x86 BCJ should help: {} vs {}", with.len(), without.len());

    // Same chain in liblzma gives (nearly) the same size — the transforms
    // agree (liblzma's streaming BCJ changes LZMA2 chunk boundaries slightly).
    let c = c_xz_compress_with_filter(&data, |f| {
        f.x86();
    });
    let c_plain = c_xz_compress(&data, 6);
    eprintln!("x86 BCJ: ours {} / liblzma {}; plain: ours {} / liblzma {}", with.len(), c.len(), without.len(), c_plain.len());
    assert!((with.len() as i64 - c.len() as i64).abs() * 200 <= c.len() as i64, "ours {} vs liblzma {}", with.len(), c.len());
    assert_eq!(ours_decode(&c), data);

    // The transform itself is byte-identical: peel LZMA2 off both raw
    // [x86, LZMA2] streams with liblzma and compare the filtered bytes.
    let mut f = Filters::new();
    f.x86().lzma2(&opts);
    let ours_raw = ours_encode(&data, 6, Format::RAW, Check::None, Some(f));
    let theirs_raw = c_raw_encode(&data, &[lzma_sys::LZMA_FILTER_X86], false, 6);
    let ours_t = c_raw_decode(&ours_raw, &[], false, 6, None, data.len());
    let theirs_t = c_raw_decode(&theirs_raw, &[], false, 6, None, data.len());
    let first_diff = ours_t.iter().zip(&theirs_t).position(|(a, b)| a != b);
    assert_eq!(first_diff, None, "x86 BCJ transform differs from liblzma at {first_diff:?} of {}", data.len());
    assert_eq!(ours_t.len(), theirs_t.len());

    // Stacked BCJ filters (x86 then ARM) are allowed too.
    let mut f = Filters::new();
    f.x86().arm().lzma2(&opts);
    let stacked = ours_encode(&data, 6, Format::XZ, Check::Crc32, Some(f));
    assert_eq!(c_xz_decompress(&stacked), data);
    assert_eq!(ours_decode(&stacked), data);
}

#[test]
fn sha256_check_cross_impl() {
    let mut data = gen_text(100_000);
    data.extend((0..40_000u32).map(|i| ((i.wrapping_mul(2654435761u32)) >> 24) as u8));
    for preset in [1u32, 6] {
        let xz = ours_encode(&data, preset, Format::XZ, Check::Sha256, None);
        assert_eq!(xz[7], 0x0A, "stream flags check id = SHA-256");
        assert_eq!(c_xz_decompress(&xz), data);
        assert_eq!(ours_decode(&xz), data);
        let theirs = c_encode_stream(
            &data,
            xz2::stream::Stream::new_easy_encoder(preset, xz2::stream::Check::Sha256).unwrap(),
        );
        assert_eq!(ours_decode(&theirs), data);

        // Corrupt one byte of the stored hash: must be rejected by both.
        // The hash is the 32 bytes right before the index (indicator 0x00).
        let mut bad = xz.clone();
        let mut probe = bad.len() - 12 - 4 - 4 - 32;
        while probe > 0 && bad[probe + 32] != 0x00 {
            probe -= 1;
        }
        bad[probe + 3] ^= 0x01;
        let mut out = Vec::new();
        let err = libcramjam::xz::decompress(&mut Cursor::new(&bad), &mut out).unwrap_err();
        assert!(err.to_string().contains("SHA-256"), "{err}");
        let mut out = Vec::new();
        assert!(xz2::read::XzDecoder::new(&bad[..]).read_to_end(&mut out).is_err());
    }
    // Other checks still fine, including None.
    for check in [Check::None, Check::Crc32, Check::Crc64] {
        let xz = ours_encode(&data, 6, Format::XZ, check, None);
        assert_eq!(c_xz_decompress(&xz), data);
    }
}

#[test]
fn invalid_chains_and_formats_are_rejected() {
    let data = gen_text(1000);
    let opts = LzmaOptions::new_preset(6).unwrap();
    let mut out = Vec::new();
    let mut try_compress = |format, f: Filters| {
        libcramjam::xz::compress(
            &mut Cursor::new(&data), &mut out, Some(6), Some(format), None::<Check>, Some(f), None::<LzmaOptions>,
        )
    };
    // LZMA1 inside .xz.
    let mut f = Filters::new();
    f.lzma1(&opts);
    assert!(try_compress(Format::XZ, f).is_err());
    // BCJ inside .lzma alone.
    let mut f = Filters::new();
    f.x86().lzma2(&opts);
    assert!(try_compress(Format::ALONE, f).is_err());
    // BCJ-only chain.
    let mut f = Filters::new();
    f.x86();
    assert!(try_compress(Format::RAW, f).is_err());
    // ALONE with an explicit LZMA1 entry works (it is the alone format's coder).
    let mut f = Filters::new();
    f.lzma1(&opts);
    let alone = ours_encode(&data, 6, Format::ALONE, Check::None, Some(f));
    assert_eq!(c_decode_stream(&alone, xz2::stream::Stream::new_lzma_decoder(u64::MAX).unwrap()), data);
}
