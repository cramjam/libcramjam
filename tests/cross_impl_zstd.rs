//! Cross-implementation tests: our pure-Rust zstd vs the C-backed `zstd` crate.

use std::io::{Cursor, Read, Write};

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
fn our_compress_c_decompress() {
    for (name, data) in [
        ("empty", vec![]),
        ("one_byte", vec![42]),
        ("text_100", gen_text(100)),
        ("text_10k", gen_text(10_000)),
        ("random_1k", gen_random(0xBEEF, 1000)),
    ] {
        let compressed = our_zstd_compress(&data, 0);
        let decompressed = c_zstd_decompress(&compressed);
        assert_eq!(
            decompressed, data,
            "ours->c failed: corpus={name}"
        );
    }
}
