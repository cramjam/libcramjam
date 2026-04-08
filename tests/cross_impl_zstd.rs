//! Cross-implementation tests: our pure-Rust zstd vs the C-backed `zstd` crate.

use std::io::{Cursor, Read};

fn gen_text(size: usize) -> Vec<u8> {
    let phrases: &[&[u8]] = &[
        b"The quick brown fox jumps over the lazy dog. ",
        b"Lorem ipsum dolor sit amet, consectetur adipiscing elit. ",
        b"fn main() { println!(\"hello world\"); }\n",
        b"pub struct Compressor { level: u32, window: Vec<u8> }\n",
    ];
    let mut data = Vec::with_capacity(size);
    for phrase in phrases.iter().cycle() {
        let take = phrase.len().min(size - data.len());
        data.extend_from_slice(&phrase[..take]);
        if data.len() >= size { break; }
    }
    data
}

fn gen_random(seed: u32, size: usize) -> Vec<u8> {
    let mut s = seed;
    (0..size).map(|_| {
        s ^= s << 13; s ^= s >> 17; s ^= s << 5;
        (s >> 16) as u8
    }).collect()
}

// --- C-backed zstd (reference) ---

fn c_zstd_compress(data: &[u8], level: i32) -> Vec<u8> {
    let mut enc = zstd::stream::read::Encoder::new(data, level).unwrap();
    let mut out = Vec::new();
    enc.read_to_end(&mut out).unwrap();
    out
}

fn c_zstd_decompress(data: &[u8]) -> Vec<u8> {
    let mut dec = zstd::stream::read::Decoder::new(data).unwrap();
    let mut out = Vec::new();
    dec.read_to_end(&mut out).unwrap();
    out
}

// --- Our pure-Rust zstd ---

fn our_zstd_compress(data: &[u8], level: i32) -> Vec<u8> {
    let mut out = Vec::new();
    libcramjam::zstd::compress(&mut Cursor::new(data), &mut out, Some(level), Some(data.len())).unwrap();
    out
}

fn our_zstd_decompress(data: &[u8]) -> Vec<u8> {
    let mut out = Vec::new();
    libcramjam::zstd::decompress(&mut Cursor::new(data), &mut out).unwrap();
    out
}

// =========================================================================
// Self-roundtrip (our compress → our decompress)
// =========================================================================

#[test]
fn self_roundtrip() {
    for (name, data) in [
        ("empty", vec![]),
        ("one_byte", vec![42]),
        ("text_100", gen_text(100)),
        ("text_10k", gen_text(10_000)),
        ("random_1k", gen_random(0xBEEF, 1000)),
    ] {
        let compressed = our_zstd_compress(&data, 0);
        let decompressed = our_zstd_decompress(&compressed);
        assert_eq!(decompressed, data, "self-roundtrip failed: {name}");
    }
}

// =========================================================================
// Cross: C-zstd compress → our decompress
// =========================================================================

#[test]
fn c_compress_our_decompress() {
    for (name, data) in [
        ("empty", vec![]),
        ("one_byte", vec![42]),
        ("text_100", gen_text(100)),
        ("text_1k", gen_text(1000)),
        ("text_10k", gen_text(10_000)),
        ("text_100k", gen_text(100_000)),
        ("random_1k", gen_random(0xDEAD, 1000)),
        ("random_10k", gen_random(0xCAFE, 10_000)),
        ("repeated", vec![0xAA; 50_000]),
        ("sequential", (0u8..=255).cycle().take(10_000).collect()),
    ] {
        for level in [1, 3, 6, 9] {
            let compressed = c_zstd_compress(&data, level);
            let decompressed = our_zstd_decompress(&compressed);
            assert_eq!(
                decompressed, data,
                "c->ours failed: corpus={name}, level={level}"
            );
        }
    }
}

// =========================================================================
// Cross: our compress → C-zstd decompress
// =========================================================================

#[test]
fn our_compress_c_decompress_binary_literals() {
    // Synthesize a corpus with > 128 distinct byte values in the literal pool
    // — this exercises the FSE-compressed Huffman weights path that direct
    // (4-bit packed) encoding can't handle.
    let mut data = Vec::new();
    for i in 0..200_000u32 {
        data.push(((i.wrapping_mul(2654435761) >> 8) & 0xFF) as u8);
    }
    let compressed = our_zstd_compress(&data, 1);
    let decompressed = c_zstd_decompress(&compressed);
    assert_eq!(decompressed, data, "binary corpus c-decode failed");
}

#[test]
fn our_compress_our_decompress_binary_literals() {
    // Same data as above but round-tripped through OUR decoder.  Catches
    // bugs in our decoder's FSE-compressed Huffman weight reader.
    let mut data = Vec::new();
    for i in 0..200_000u32 {
        data.push(((i.wrapping_mul(2654435761) >> 8) & 0xFF) as u8);
    }
    let compressed = our_zstd_compress(&data, 1);
    let decompressed = our_zstd_decompress(&compressed);
    assert_eq!(decompressed, data, "binary corpus our-decode failed");
}

fn read_src_dir() -> Vec<u8> {
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
    use std::str::FromStr;
    let mut bytes = read_dir_files(std::path::PathBuf::from_str("./src").unwrap());
    while bytes.len() < 5_000_000 {
        bytes.extend(bytes.clone());
    }
    bytes
}

#[test]
fn our_roundtrip_src_dir() {
    let bytes = read_src_dir();
    let compressed = our_zstd_compress(&bytes, 1);
    let decompressed = our_zstd_decompress(&compressed);
    assert_eq!(decompressed.len(), bytes.len(), "size mismatch");
    assert_eq!(decompressed, bytes, "src corpus our-decode failed");
}

#[test]
fn our_compress_c_decompress_src_dir() {
    // Same data through C decoder — if this passes but our_roundtrip_src_dir
    // fails, the bug is in OUR decoder not the encoder.
    let bytes = read_src_dir();
    let compressed = our_zstd_compress(&bytes, 1);
    let decompressed = c_zstd_decompress(&compressed);
    assert_eq!(decompressed.len(), bytes.len(), "size mismatch");
    assert_eq!(decompressed, bytes, "src corpus c-decode failed");
}

#[test]
fn our_compress_c_decompress_single_file() {
    // Read just one source file to find a smaller failing case.
    let bytes = std::fs::read("./src/zstd_impl/encode.rs").unwrap();
    let compressed = our_zstd_compress(&bytes, 1);
    let decompressed = c_zstd_decompress(&compressed);
    assert_eq!(decompressed, bytes, "single file c-decode failed");
}

#[test]
fn our_compress_our_decompress_single_file() {
    let bytes = std::fs::read("./src/zstd_impl/encode.rs").unwrap();
    let compressed = our_zstd_compress(&bytes, 1);
    let decompressed = our_zstd_decompress(&compressed);
    assert_eq!(decompressed, bytes, "single file ours-decode failed");
}

#[test]
fn our_compress_c_decompress() {
    for (name, data) in [
        ("empty", vec![]),
        ("one_byte", vec![42]),
        ("text_100", gen_text(100)),
        ("text_10k", gen_text(10_000)),
        ("text_100k", gen_text(100_000)),
        ("text_130k", gen_text(130_000)), // crosses 128KB block boundary
        ("text_300k", gen_text(300_000)),
        ("text_1m", gen_text(1_000_000)),
        ("random_1k", gen_random(0xBEEF, 1000)),
        ("random_100k", gen_random(0xBEEF, 100_000)),
        ("repeated_500k", vec![0xAAu8; 500_000]),
    ] {
        for level in [1, 3, 6, 9] {
            let compressed = our_zstd_compress(&data, level);
            let decompressed = c_zstd_decompress(&compressed);
            assert_eq!(
                decompressed, data,
                "ours->c failed: corpus={name}, level={level}"
            );
        }
    }
}

#[test]
fn our_roundtrip_large() {
    // Round-trip large inputs through our encoder + decoder.
    for (name, data) in [
        ("text_1m", gen_text(1_000_000)),
        ("text_5m", gen_text(5_000_000)),
        ("repeated_2m", vec![0xAAu8; 2_000_000]),
    ] {
        for level in [1, 3, 6, 9] {
            let compressed = our_zstd_compress(&data, level);
            let decompressed = our_zstd_decompress(&compressed);
            assert_eq!(
                decompressed, data,
                "ours->ours failed: corpus={name}, level={level}"
            );
        }
    }
}
