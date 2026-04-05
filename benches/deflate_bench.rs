//! Benchmarks comparing our pure-Rust deflate/gzip/zlib against flate2.
//!
//! Run with: cargo bench --bench deflate_bench

use criterion::{
    criterion_group, criterion_main, BenchmarkId, Criterion, Throughput,
};
use std::io::{Cursor, Read, Write};

// ---------------------------------------------------------------------------
// Corpus helpers
// ---------------------------------------------------------------------------

/// Deterministic text-like data (repeating phrases, good compressibility).
fn gen_text(size: usize) -> Vec<u8> {
    let phrases: &[&[u8]] = &[
        b"The quick brown fox jumps over the lazy dog. ",
        b"Lorem ipsum dolor sit amet, consectetur adipiscing elit. ",
        b"fn main() { println!(\"hello world\"); }\n",
        b"pub struct Compressor { level: u32, window: Vec<u8> }\n",
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

/// Pseudo-random data (poor compressibility, stresses stored/literal paths).
fn gen_random(size: usize) -> Vec<u8> {
    let mut s: u32 = 0xDEAD_BEEF;
    (0..size)
        .map(|_| {
            s ^= s << 13;
            s ^= s >> 17;
            s ^= s << 5;
            (s >> 16) as u8
        })
        .collect()
}

// ---------------------------------------------------------------------------
// Wrapper helpers — keep closure bodies tiny for criterion
// ---------------------------------------------------------------------------

// ours
fn ours_deflate_compress(data: &[u8], level: u32) -> Vec<u8> {
    let mut out = Vec::with_capacity(data.len());
    libcramjam::deflate::compress(&mut Cursor::new(data), &mut out, Some(level)).unwrap();
    out
}
fn ours_deflate_decompress(data: &[u8]) -> Vec<u8> {
    let mut out = Vec::with_capacity(data.len() * 2);
    libcramjam::deflate::decompress(&mut Cursor::new(data), &mut out).unwrap();
    out
}
fn ours_gzip_compress(data: &[u8], level: u32) -> Vec<u8> {
    let mut out = Vec::with_capacity(data.len());
    libcramjam::gzip::compress(&mut Cursor::new(data), &mut out, Some(level)).unwrap();
    out
}
fn ours_gzip_decompress(data: &[u8]) -> Vec<u8> {
    let mut out = Vec::with_capacity(data.len() * 2);
    libcramjam::gzip::decompress(&mut Cursor::new(data), &mut out).unwrap();
    out
}
fn ours_zlib_compress(data: &[u8], level: u32) -> Vec<u8> {
    let mut out = Vec::with_capacity(data.len());
    libcramjam::zlib::compress(&mut Cursor::new(data), &mut out, Some(level)).unwrap();
    out
}
fn ours_zlib_decompress(data: &[u8]) -> Vec<u8> {
    let mut out = Vec::with_capacity(data.len() * 2);
    libcramjam::zlib::decompress(&mut Cursor::new(data), &mut out).unwrap();
    out
}

// flate2
fn f2_deflate_compress(data: &[u8], level: u32) -> Vec<u8> {
    let mut e =
        flate2::write::DeflateEncoder::new(Vec::with_capacity(data.len()), flate2::Compression::new(level));
    e.write_all(data).unwrap();
    e.finish().unwrap()
}
fn f2_deflate_decompress(data: &[u8]) -> Vec<u8> {
    let mut d = flate2::read::DeflateDecoder::new(data);
    let mut out = Vec::with_capacity(data.len() * 2);
    d.read_to_end(&mut out).unwrap();
    out
}
fn f2_gzip_compress(data: &[u8], level: u32) -> Vec<u8> {
    let mut e =
        flate2::write::GzEncoder::new(Vec::with_capacity(data.len()), flate2::Compression::new(level));
    e.write_all(data).unwrap();
    e.finish().unwrap()
}
fn f2_gzip_decompress(data: &[u8]) -> Vec<u8> {
    let mut d = flate2::read::MultiGzDecoder::new(data);
    let mut out = Vec::with_capacity(data.len() * 2);
    d.read_to_end(&mut out).unwrap();
    out
}
fn f2_zlib_compress(data: &[u8], level: u32) -> Vec<u8> {
    let mut e =
        flate2::write::ZlibEncoder::new(Vec::with_capacity(data.len()), flate2::Compression::new(level));
    e.write_all(data).unwrap();
    e.finish().unwrap()
}
fn f2_zlib_decompress(data: &[u8]) -> Vec<u8> {
    let mut d = flate2::read::ZlibDecoder::new(data);
    let mut out = Vec::with_capacity(data.len() * 2);
    d.read_to_end(&mut out).unwrap();
    out
}

// =========================================================================
// Benchmark groups
// =========================================================================

fn bench_compress(c: &mut Criterion) {
    let corpora: Vec<(&str, Vec<u8>)> = vec![
        ("text_10k", gen_text(10_000)),
        ("text_100k", gen_text(100_000)),
        ("random_100k", gen_random(100_000)),
    ];
    let levels = [1, 6, 9];

    for (corpus_name, data) in &corpora {
        let mut group = c.benchmark_group(format!("compress/{corpus_name}"));
        group.throughput(Throughput::Bytes(data.len() as u64));

        for &level in &levels {
            group.bench_with_input(
                BenchmarkId::new("ours_deflate", level),
                &level,
                |b, &l| b.iter(|| ours_deflate_compress(data, l)),
            );
            group.bench_with_input(
                BenchmarkId::new("flate2_deflate", level),
                &level,
                |b, &l| b.iter(|| f2_deflate_compress(data, l)),
            );
            group.bench_with_input(
                BenchmarkId::new("ours_gzip", level),
                &level,
                |b, &l| b.iter(|| ours_gzip_compress(data, l)),
            );
            group.bench_with_input(
                BenchmarkId::new("flate2_gzip", level),
                &level,
                |b, &l| b.iter(|| f2_gzip_compress(data, l)),
            );
            group.bench_with_input(
                BenchmarkId::new("ours_zlib", level),
                &level,
                |b, &l| b.iter(|| ours_zlib_compress(data, l)),
            );
            group.bench_with_input(
                BenchmarkId::new("flate2_zlib", level),
                &level,
                |b, &l| b.iter(|| f2_zlib_compress(data, l)),
            );
        }
        group.finish();
    }
}

fn bench_decompress(c: &mut Criterion) {
    let corpora: Vec<(&str, Vec<u8>)> = vec![
        ("text_10k", gen_text(10_000)),
        ("text_100k", gen_text(100_000)),
        ("random_100k", gen_random(100_000)),
    ];
    // Decompress data compressed at level 6 (the common case).
    let level = 6u32;

    for (corpus_name, data) in &corpora {
        let mut group = c.benchmark_group(format!("decompress/{corpus_name}"));
        group.throughput(Throughput::Bytes(data.len() as u64));

        // Pre-compress with each impl at level 6
        let ours_def = ours_deflate_compress(data, level);
        let f2_def = f2_deflate_compress(data, level);
        let ours_gz = ours_gzip_compress(data, level);
        let f2_gz = f2_gzip_compress(data, level);
        let ours_zl = ours_zlib_compress(data, level);
        let f2_zl = f2_zlib_compress(data, level);

        // Deflate: decompress our output
        group.bench_function("ours_deflate/ours_data", |b| {
            b.iter(|| ours_deflate_decompress(&ours_def))
        });
        // Deflate: decompress flate2 output (tests our decoder on their encoder output)
        group.bench_function("ours_deflate/f2_data", |b| {
            b.iter(|| ours_deflate_decompress(&f2_def))
        });
        group.bench_function("flate2_deflate/f2_data", |b| {
            b.iter(|| f2_deflate_decompress(&f2_def))
        });
        group.bench_function("flate2_deflate/ours_data", |b| {
            b.iter(|| f2_deflate_decompress(&ours_def))
        });

        // Gzip
        group.bench_function("ours_gzip/ours_data", |b| {
            b.iter(|| ours_gzip_decompress(&ours_gz))
        });
        group.bench_function("ours_gzip/f2_data", |b| {
            b.iter(|| ours_gzip_decompress(&f2_gz))
        });
        group.bench_function("flate2_gzip/f2_data", |b| {
            b.iter(|| f2_gzip_decompress(&f2_gz))
        });
        group.bench_function("flate2_gzip/ours_data", |b| {
            b.iter(|| f2_gzip_decompress(&ours_gz))
        });

        // Zlib
        group.bench_function("ours_zlib/ours_data", |b| {
            b.iter(|| ours_zlib_decompress(&ours_zl))
        });
        group.bench_function("ours_zlib/f2_data", |b| {
            b.iter(|| ours_zlib_decompress(&f2_zl))
        });
        group.bench_function("flate2_zlib/f2_data", |b| {
            b.iter(|| f2_zlib_decompress(&f2_zl))
        });
        group.bench_function("flate2_zlib/ours_data", |b| {
            b.iter(|| f2_zlib_decompress(&ours_zl))
        });

        group.finish();
    }
}

/// Quick ratio check: print compressed sizes side-by-side.
fn bench_compression_ratio(c: &mut Criterion) {
    let corpora: Vec<(&str, Vec<u8>)> = vec![
        ("text_100k", gen_text(100_000)),
        ("random_100k", gen_random(100_000)),
    ];

    let mut group = c.benchmark_group("compression_ratio");
    // This group doesn't need repeated timing — just a single pass to report sizes.
    group.sample_size(10);

    for (corpus_name, data) in &corpora {
        for level in [1, 6, 9] {
            let ours = ours_deflate_compress(&data, level);
            let f2 = f2_deflate_compress(&data, level);
            let ratio_ours = ours.len() as f64 / data.len() as f64;
            let ratio_f2 = f2.len() as f64 / data.len() as f64;
            eprintln!(
                "[ratio] {corpus_name} level={level}: ours={} ({:.1}%) flate2={} ({:.1}%) delta={:+.1}%",
                ours.len(),
                ratio_ours * 100.0,
                f2.len(),
                ratio_f2 * 100.0,
                (ratio_ours - ratio_f2) * 100.0,
            );
            // Bench a no-op to make criterion happy — the real output is the eprintln above.
            group.bench_function(
                format!("{corpus_name}/level{level}/ours"),
                |b| b.iter(|| ours_deflate_compress(&data, level)),
            );
            group.bench_function(
                format!("{corpus_name}/level{level}/flate2"),
                |b| b.iter(|| f2_deflate_compress(&data, level)),
            );
        }
    }
    group.finish();
}

criterion_group!(benches, bench_compress, bench_decompress, bench_compression_ratio);
criterion_main!(benches);
