//! Cross-implementation integration tests.
//!
//! Verifies that our pure-Rust deflate/gzip/zlib implementation is
//! byte-stream-compatible with flate2 (which uses miniz_oxide).
//! Tests both directions: ours -> flate2 and flate2 -> ours.

use std::io::{Cursor, Read, Write};

// ---------------------------------------------------------------------------
// Test data helpers
// ---------------------------------------------------------------------------

/// Deterministic PRNG (xorshift32).
fn gen_prng(seed: u32, size: usize) -> Vec<u8> {
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

/// Text-like data with varying patterns.
fn gen_text(size: usize) -> Vec<u8> {
    let phrases: &[&[u8]] = &[
        b"The quick brown fox jumps over the lazy dog. ",
        b"Lorem ipsum dolor sit amet, consectetur adipiscing elit. ",
        b"fn main() { println!(\"hello world\"); }\n",
        b"pub struct Compressor { level: u32, window: Vec<u8> }\n",
        b"impl Write for BitWriter { fn write(&mut self, buf: &[u8]) -> io::Result<usize> { Ok(0) } }\n",
        b"#[test] fn roundtrip() { assert_eq!(decompress(compress(data)), data); }\n",
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

/// Read every file under `dir` recursively.
fn read_dir_files(dir: &std::path::Path) -> Vec<u8> {
    let mut out = Vec::new();
    let Ok(entries) = std::fs::read_dir(dir) else { return out };
    let mut entries: Vec<_> = entries.filter_map(|e| e.ok()).collect();
    entries.sort_by_key(|e| e.path());
    for entry in entries {
        let ft = entry.file_type().unwrap();
        if ft.is_file() {
            out.extend(std::fs::read(entry.path()).unwrap());
        } else if ft.is_dir() {
            out.extend(read_dir_files(&entry.path()));
        }
    }
    out
}

fn test_corpora() -> Vec<(&'static str, Vec<u8>)> {
    let src = read_dir_files(std::path::Path::new("./src"));
    let mut large = src.clone();
    while large.len() < 2_000_000 {
        large.extend(large.clone());
    }
    vec![
        ("empty", vec![]),
        ("one_byte", vec![42]),
        ("small_text", b"Hello, world!".to_vec()),
        ("repeated", vec![0xAA; 100_000]),
        ("sequential", (0u8..=255).cycle().take(65_536).collect()),
        ("source_files", src),
        ("large_2mb", large),
    ]
}

// ---------------------------------------------------------------------------
// flate2 reference helpers
// ---------------------------------------------------------------------------

fn f2_gz_c(data: &[u8], level: u32) -> Vec<u8> {
    let mut e = flate2::write::GzEncoder::new(Vec::new(), flate2::Compression::new(level));
    e.write_all(data).unwrap();
    e.finish().unwrap()
}
fn f2_gz_d(data: &[u8]) -> Vec<u8> {
    let mut d = flate2::read::MultiGzDecoder::new(data);
    let mut o = Vec::new();
    d.read_to_end(&mut o).unwrap();
    o
}
fn f2_def_c(data: &[u8], level: u32) -> Vec<u8> {
    let mut e = flate2::write::DeflateEncoder::new(Vec::new(), flate2::Compression::new(level));
    e.write_all(data).unwrap();
    e.finish().unwrap()
}
fn f2_def_d(data: &[u8]) -> Vec<u8> {
    let mut d = flate2::read::DeflateDecoder::new(data);
    let mut o = Vec::new();
    d.read_to_end(&mut o).unwrap();
    o
}
fn f2_zl_c(data: &[u8], level: u32) -> Vec<u8> {
    let mut e = flate2::write::ZlibEncoder::new(Vec::new(), flate2::Compression::new(level));
    e.write_all(data).unwrap();
    e.finish().unwrap()
}
fn f2_zl_d(data: &[u8]) -> Vec<u8> {
    let mut d = flate2::read::ZlibDecoder::new(data);
    let mut o = Vec::new();
    d.read_to_end(&mut o).unwrap();
    o
}

// ---------------------------------------------------------------------------
// libcramjam helpers
// ---------------------------------------------------------------------------

fn our_gz_c(d: &[u8], l: u32) -> Vec<u8> {
    let mut o = Vec::new();
    libcramjam::gzip::compress(&mut Cursor::new(d), &mut o, Some(l)).unwrap();
    o
}
fn our_gz_d(d: &[u8]) -> Vec<u8> {
    let mut o = Vec::new();
    libcramjam::gzip::decompress(&mut Cursor::new(d), &mut o).unwrap();
    o
}
fn our_def_c(d: &[u8], l: u32) -> Vec<u8> {
    let mut o = Vec::new();
    libcramjam::deflate::compress(&mut Cursor::new(d), &mut o, Some(l)).unwrap();
    o
}
fn our_def_d(d: &[u8]) -> Vec<u8> {
    let mut o = Vec::new();
    libcramjam::deflate::decompress(&mut Cursor::new(d), &mut o).unwrap();
    o
}
fn our_zl_c(d: &[u8], l: u32) -> Vec<u8> {
    let mut o = Vec::new();
    libcramjam::zlib::compress(&mut Cursor::new(d), &mut o, Some(l)).unwrap();
    o
}
fn our_zl_d(d: &[u8]) -> Vec<u8> {
    let mut o = Vec::new();
    libcramjam::zlib::decompress(&mut Cursor::new(d), &mut o).unwrap();
    o
}

// =========================================================================
// Multi-corpus tests (levels 0, 1, 6, 9)
// =========================================================================

#[test]
fn gzip_ours_to_flate2() {
    for (name, data) in test_corpora() {
        for level in [0, 1, 6, 9] {
            assert_eq!(f2_gz_d(&our_gz_c(&data, level)), data,
                "gzip ours->flate2: {name} level={level}");
        }
    }
}

#[test]
fn gzip_flate2_to_ours() {
    for (name, data) in test_corpora() {
        for level in [0, 1, 6, 9] {
            assert_eq!(our_gz_d(&f2_gz_c(&data, level)), data,
                "gzip flate2->ours: {name} level={level}");
        }
    }
}

#[test]
fn deflate_ours_to_flate2() {
    for (name, data) in test_corpora() {
        for level in [0, 1, 6, 9] {
            assert_eq!(f2_def_d(&our_def_c(&data, level)), data,
                "deflate ours->flate2: {name} level={level}");
        }
    }
}

#[test]
fn deflate_flate2_to_ours() {
    for (name, data) in test_corpora() {
        for level in [0, 1, 6, 9] {
            assert_eq!(our_def_d(&f2_def_c(&data, level)), data,
                "deflate flate2->ours: {name} level={level}");
        }
    }
}

#[test]
fn zlib_ours_to_flate2() {
    for (name, data) in test_corpora() {
        for level in [0, 1, 6, 9] {
            assert_eq!(f2_zl_d(&our_zl_c(&data, level)), data,
                "zlib ours->flate2: {name} level={level}");
        }
    }
}

#[test]
fn zlib_flate2_to_ours() {
    for (name, data) in test_corpora() {
        for level in [0, 1, 6, 9] {
            assert_eq!(our_zl_d(&f2_zl_c(&data, level)), data,
                "zlib flate2->ours: {name} level={level}");
        }
    }
}

// =========================================================================
// All levels 0-9 with deterministic data (no source-file dependency)
// =========================================================================

#[test]
fn deflate_all_levels_deterministic() {
    let corpora = [
        gen_text(100),
        gen_text(50_000),
        gen_text(200_000),
        gen_prng(0xCAFE, 50_000),
        vec![0u8; 100_000],
    ];
    for data in &corpora {
        for level in 0..=9u32 {
            assert_eq!(f2_def_d(&our_def_c(data, level)), *data,
                "deflate ours->flate2 level={level} len={}", data.len());
            assert_eq!(our_def_d(&f2_def_c(data, level)), *data,
                "deflate flate2->ours level={level} len={}", data.len());
        }
    }
}

#[test]
fn gzip_all_levels_deterministic() {
    let corpora = [gen_text(100), gen_text(50_000), gen_prng(0xBEEF, 50_000)];
    for data in &corpora {
        for level in 0..=9u32 {
            assert_eq!(f2_gz_d(&our_gz_c(data, level)), *data,
                "gzip ours->flate2 level={level} len={}", data.len());
            assert_eq!(our_gz_d(&f2_gz_c(data, level)), *data,
                "gzip flate2->ours level={level} len={}", data.len());
        }
    }
}

#[test]
fn zlib_all_levels_deterministic() {
    let corpora = [gen_text(100), gen_text(50_000), gen_prng(0xDEAD, 50_000)];
    for data in &corpora {
        for level in 0..=9u32 {
            assert_eq!(f2_zl_d(&our_zl_c(data, level)), *data,
                "zlib ours->flate2 level={level} len={}", data.len());
            assert_eq!(our_zl_d(&f2_zl_c(data, level)), *data,
                "zlib flate2->ours level={level} len={}", data.len());
        }
    }
}

// =========================================================================
// Concatenated gzip streams
// =========================================================================

#[test]
fn gzip_concat_ours_then_flate2() {
    let a = b"first chunk of concatenation test";
    let b = b"second chunk of concatenation test";
    let mut stream = our_gz_c(a, 6);
    stream.extend(our_gz_c(b, 6));
    let mut expected = a.to_vec();
    expected.extend_from_slice(b);
    assert_eq!(f2_gz_d(&stream), expected);
}

#[test]
fn gzip_concat_flate2_then_ours() {
    let a = b"first chunk of concatenation test";
    let b = b"second chunk of concatenation test";
    let mut stream = f2_gz_c(a, 6);
    stream.extend(f2_gz_c(b, 6));
    let mut expected = a.to_vec();
    expected.extend_from_slice(b);
    assert_eq!(our_gz_d(&stream), expected);
}

#[test]
fn gzip_concat_mixed_impls() {
    let a = b"hello from flate2";
    let b = b"hello from pure rust";
    let mut stream = f2_gz_c(a, 6);
    stream.extend(our_gz_c(b, 6));
    let mut expected = a.to_vec();
    expected.extend_from_slice(b);
    assert_eq!(our_gz_d(&stream), expected);
    assert_eq!(f2_gz_d(&stream), expected);
}

// =========================================================================
// Edge cases
// =========================================================================

#[test]
fn all_byte_values() {
    let data: Vec<u8> = (0u8..=255).collect();
    for level in [1, 6, 9] {
        assert_eq!(f2_gz_d(&our_gz_c(&data, level)), data);
        assert_eq!(f2_def_d(&our_def_c(&data, level)), data);
        assert_eq!(f2_zl_d(&our_zl_c(&data, level)), data);
    }
}

#[test]
fn highly_compressible() {
    let data = vec![0u8; 1_000_000];
    for level in [0, 1, 6, 9] {
        assert_eq!(f2_gz_d(&our_gz_c(&data, level)), data, "level={level}");
        assert_eq!(f2_zl_d(&our_zl_c(&data, level)), data, "level={level}");
    }
}

#[test]
fn incompressible_random() {
    let data = gen_prng(0xDEAD_BEEF, 50_000);
    for level in [0, 1, 6, 9] {
        assert_eq!(f2_gz_d(&our_gz_c(&data, level)), data);
        assert_eq!(our_gz_d(&f2_gz_c(&data, level)), data);
    }
}

#[test]
fn compress_bound_sufficient() {
    for (name, data) in test_corpora() {
        let bound = libcramjam::deflate::compress_bound(data.len());
        for level in [0, 1, 6, 9] {
            let c = our_def_c(&data, level);
            assert!(c.len() <= bound,
                "compress_bound too small: {name} level={level} bound={bound} actual={}", c.len());
        }
    }
}

// =========================================================================
// Fuzz-like: many seeds × sizes × levels
// =========================================================================

#[test]
fn fuzz_many_seeds() {
    for seed in 0..50u32 {
        let size = 1000 + (seed as usize * 3333);
        let data = gen_prng(seed.wrapping_mul(2654435761), size);
        for level in [1, 6, 9] {
            assert_eq!(f2_def_d(&our_def_c(&data, level)), data,
                "fuzz ours->flate2 seed={seed} size={size} level={level}");
            assert_eq!(our_def_d(&f2_def_c(&data, level)), data,
                "fuzz flate2->ours seed={seed} size={size} level={level}");
        }
    }
}
