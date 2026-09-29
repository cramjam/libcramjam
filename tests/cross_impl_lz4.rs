//! Cross-implementation tests: our pure-Rust lz4 against the C-backed `lz4` crate.

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
        if data.len() >= size {
            break;
        }
    }
    data
}

fn gen_random(seed: u32, size: usize) -> Vec<u8> {
    let mut s = seed;
    (0..size)
        .map(|_| {
            s ^= s << 13;
            s ^= s >> 17;
            s ^= s << 5;
            (s >> 16) as u8
        })
        .collect()
}

// --- C-backed lz4 (reference) ---

fn c_lz4_compress(data: &[u8]) -> Vec<u8> {
    let mut enc = lz4::EncoderBuilder::new()
        .level(4)
        .auto_flush(true)
        .build(Vec::new())
        .unwrap();
    use std::io::Write;
    enc.write_all(data).unwrap();
    let (out, r) = enc.finish();
    r.unwrap();
    out
}

fn c_lz4_decompress(data: &[u8]) -> Vec<u8> {
    let mut dec = lz4::Decoder::new(data).unwrap();
    let mut out = Vec::new();
    dec.read_to_end(&mut out).unwrap();
    out
}

// --- C-backed lz4 at a specific level ---

fn c_lz4_compress_at(data: &[u8], level: u32) -> Vec<u8> {
    let mut enc = lz4::EncoderBuilder::new()
        .level(level)
        .auto_flush(true)
        .build(Vec::new())
        .unwrap();
    use std::io::Write;
    enc.write_all(data).unwrap();
    let (out, r) = enc.finish();
    r.unwrap();
    out
}

// --- Our pure-Rust lz4 ---

fn our_lz4_compress(data: &[u8]) -> Vec<u8> {
    let mut out = Vec::new();
    libcramjam::lz4::compress(&mut Cursor::new(data), &mut out, None).unwrap();
    out
}

fn our_lz4_compress_at(data: &[u8], level: u32) -> Vec<u8> {
    let mut out = Vec::new();
    libcramjam::lz4::compress(&mut Cursor::new(data), &mut out, Some(level)).unwrap();
    out
}

fn our_lz4_decompress(data: &[u8]) -> Vec<u8> {
    let mut out = Vec::new();
    libcramjam::lz4::decompress(&mut Cursor::new(data), &mut out).unwrap();
    out
}

// =========================================================================
// Self-roundtrip
// =========================================================================

#[test]
fn self_roundtrip() {
    for (name, data) in [
        ("empty", vec![]),
        ("one_byte", vec![42]),
        ("text_100", gen_text(100)),
        ("text_10k", gen_text(10_000)),
        ("text_100k", gen_text(100_000)),
        ("text_1m", gen_text(1_000_000)),
        ("random_100k", gen_random(0xBEEF, 100_000)),
        ("repeated", vec![0xAA; 50_000]),
    ] {
        let compressed = our_lz4_compress(&data);
        let decompressed = our_lz4_decompress(&compressed);
        assert_eq!(decompressed, data, "self-roundtrip failed: {name}");
    }
}

// =========================================================================
// Cross: C lz4 → our decompress
// =========================================================================

#[test]
fn c_compress_our_decompress() {
    for (name, data) in [
        ("empty", vec![]),
        ("one_byte", vec![42]),
        ("text_100", gen_text(100)),
        ("text_10k", gen_text(10_000)),
        ("text_100k", gen_text(100_000)),
        ("text_1m", gen_text(1_000_000)),
        ("random_10k", gen_random(0xCAFE, 10_000)),
        ("random_100k", gen_random(0xCAFE, 100_000)),
        ("repeated", vec![0xAA; 50_000]),
    ] {
        let compressed = c_lz4_compress(&data);
        let decompressed = our_lz4_decompress(&compressed);
        assert_eq!(decompressed, data, "c→ours failed: corpus={name}");
    }
}

// =========================================================================
// Cross: our compress → C lz4 decompress
// =========================================================================

#[test]
fn our_compress_c_decompress() {
    for (name, data) in [
        ("empty", vec![]),
        ("one_byte", vec![42]),
        ("text_100", gen_text(100)),
        ("text_10k", gen_text(10_000)),
        ("text_100k", gen_text(100_000)),
        ("text_1m", gen_text(1_000_000)),
        ("random_10k", gen_random(0xBEEF, 10_000)),
        ("random_100k", gen_random(0xBEEF, 100_000)),
        ("repeated_500k", vec![0xAA; 500_000]),
    ] {
        let compressed = our_lz4_compress(&data);
        let decompressed = c_lz4_decompress(&compressed);
        assert_eq!(decompressed, data, "ours→c failed: corpus={name}");
    }
}

// =========================================================================
// Cross: level variation — both fast (L1) and HC (L3, L6, L9, L12)
// =========================================================================

#[test]
fn our_compress_c_decompress_all_levels() {
    let data = gen_text(100_000);
    for level in [1, 3, 6, 9, 12] {
        let compressed = our_lz4_compress_at(&data, level);
        let decompressed = c_lz4_decompress(&compressed);
        assert_eq!(
            decompressed, data,
            "ours(L{level})→c failed: text_100k"
        );
    }
}

#[test]
fn c_compress_our_decompress_all_levels() {
    let data = gen_text(100_000);
    for level in [1, 4, 6, 9, 12] {
        let compressed = c_lz4_compress_at(&data, level);
        let decompressed = our_lz4_decompress(&compressed);
        assert_eq!(
            decompressed, data,
            "c(L{level})→ours failed: text_100k"
        );
    }
}

#[test]
fn self_roundtrip_all_levels() {
    let data = gen_text(100_000);
    for level in [1, 3, 6, 9, 12] {
        let compressed = our_lz4_compress_at(&data, level);
        let decompressed = our_lz4_decompress(&compressed);
        assert_eq!(
            decompressed, data,
            "ours(L{level})→ours failed: text_100k"
        );
    }
}

/// Frame options the Python wrapper exposes: independent blocks and the
/// xxhash32 content checksum. The C decoder verifies both.
#[cfg(feature = "lz4-pure")] // uses a pure-Rust test hook
#[test]
fn frame_options_content_checksum_and_independent_blocks() {
    use std::io::Read;
    let data: Vec<u8> = (0..300_000u32).map(|i| ((i * 7919) % 251) as u8 ^ (i >> 12) as u8).collect();
    for &(linked, checksum) in &[(true, true), (false, true), (false, false), (true, false)] {
        for level in [1u32, 2, 4, 9] {
            let frame = libcramjam::lz4_frame_opts_for_tests(&data, Some(level), linked, checksum);
            let mut out = Vec::new();
            lz4::Decoder::new(&frame[..]).unwrap().read_to_end(&mut out).unwrap();
            assert_eq!(out, data, "linked={linked} checksum={checksum} level={level}");
            let mut ours = Vec::new();
            libcramjam::lz4::decompress(&mut std::io::Cursor::new(&frame), &mut ours).unwrap();
            assert_eq!(ours, data);
        }
    }
}
