//! Benchmarks comparing our pure-Rust deflate/gzip/zlib against flate2.
//!
//! Run with: cargo bench --bench deflate_bench
//!
//! Inputs come from the shared corpus under `benches/data/` (see `common.rs`).

use criterion::{
    criterion_group, criterion_main, BenchmarkId, Criterion, Throughput,
};
use std::io::{Cursor, Read, Write};

#[path = "common.rs"]
mod common;

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

const LEVELS: &[u32] = &[1, 6, 9];

// =========================================================================
// Benchmark groups
// =========================================================================

fn bench_compress(c: &mut Criterion) {
    for (corpus_name, data) in common::load_bench_subset() {
        let mut group = c.benchmark_group(format!("compress/{corpus_name}"));
        group.throughput(Throughput::Bytes(data.len() as u64));
        group.sample_size(20);

        for &level in LEVELS {
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
    let level = 6u32;

    for (corpus_name, data) in common::load_bench_subset() {
        let mut group = c.benchmark_group(format!("decompress/{corpus_name}"));
        group.throughput(Throughput::Bytes(data.len() as u64));
        group.sample_size(20);

        // Pre-compress with each impl at level 6.
        let ours_def = ours_deflate_compress(data, level);
        let f2_def = f2_deflate_compress(data, level);
        let ours_gz = ours_gzip_compress(data, level);
        let f2_gz = f2_gzip_compress(data, level);
        let ours_zl = ours_zlib_compress(data, level);
        let f2_zl = f2_zlib_compress(data, level);

        // Deflate
        group.bench_function("ours_deflate/ours_data", |b| {
            b.iter(|| ours_deflate_decompress(&ours_def))
        });
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

/// Quick ratio check: print compressed sizes side-by-side for the deflate
/// codec across the bench corpus.
fn bench_compression_ratio(c: &mut Criterion) {
    let mut group = c.benchmark_group("compression_ratio");
    group.sample_size(10);

    for (corpus_name, data) in common::load_bench_subset() {
        for &level in LEVELS {
            let ours = ours_deflate_compress(data, level);
            let f2 = f2_deflate_compress(data, level);
            let ratio_ours = ours.len() as f64 / data.len() as f64;
            let ratio_f2 = f2.len() as f64 / data.len() as f64;
            eprintln!(
                "[ratio] {corpus_name} L{level}: ours={} ({:.1}%) flate2={} ({:.1}%) delta={:+.2}pp",
                ours.len(),
                ratio_ours * 100.0,
                f2.len(),
                ratio_f2 * 100.0,
                (ratio_ours - ratio_f2) * 100.0,
            );
            group.bench_function(
                format!("{corpus_name}/L{level}/ours"),
                |b| b.iter(|| ours_deflate_compress(data, level)),
            );
            group.bench_function(
                format!("{corpus_name}/L{level}/flate2"),
                |b| b.iter(|| f2_deflate_compress(data, level)),
            );
        }
    }
    group.finish();
}

// =========================================================================
// Targeted overhead micro-benchmarks (kept from the original bench file).
// These intentionally use a single small corpus file (`alice29`) to isolate
// CRC32 / wrapper costs from the main throughput numbers above.
// =========================================================================

/// Isolate: gzip decompress WITHOUT the final write_all copy.
/// Measures inflate + CRC32 only.
fn bench_gzip_inflate_only(c: &mut Criterion) {
    let data = common::load("alice29");
    let gz = f2_gzip_compress(data, 6);

    let mut group = c.benchmark_group("gzip_overhead");
    group.throughput(Throughput::Bytes(data.len() as u64));

    // Full gzip::decompress (includes write_all copy).
    group.bench_function("full_gzip_decompress", |b| {
        b.iter(|| {
            let mut out = Vec::with_capacity(data.len());
            libcramjam::gzip::decompress(&mut Cursor::new(gz.as_slice()), &mut out).unwrap();
            out
        })
    });

    // Raw deflate decompress (no checksum, no gzip wrapper).
    group.bench_function("raw_deflate_decompress", |b| {
        // Strip gzip header/footer to get raw deflate.
        let raw = &gz[10..gz.len() - 8]; // skip 10-byte header, 8-byte footer
        b.iter(|| {
            let mut out = Vec::with_capacity(data.len());
            libcramjam::deflate::decompress(&mut Cursor::new(raw), &mut out).unwrap();
            out
        })
    });

    // Measure just a memcpy of the same size.
    group.bench_function("memcpy", |b| {
        let src = vec![0u8; data.len()];
        b.iter(|| {
            let mut dst = Vec::with_capacity(data.len());
            dst.extend_from_slice(&src);
            dst
        })
    });

    group.finish();
}

fn bench_crc32_raw(c: &mut Criterion) {
    let data = common::load("alice29");
    let mut group = c.benchmark_group("crc32_raw");
    group.throughput(Throughput::Bytes(data.len() as u64));
    let gz = f2_gzip_compress(data, 6);
    let raw = &gz[10..gz.len() - 8];

    group.bench_function("our_inflate_only", |b| {
        b.iter(|| {
            let mut out = Vec::with_capacity(data.len());
            libcramjam::deflate::decompress(&mut Cursor::new(raw), &mut out).unwrap();
            out
        })
    });
    group.bench_function("our_gzip_full", |b| {
        b.iter(|| {
            let mut out = Vec::with_capacity(data.len());
            libcramjam::gzip::decompress(&mut Cursor::new(gz.as_slice()), &mut out).unwrap();
            out
        })
    });
    group.bench_function("flate2_gzip_full", |b| {
        b.iter(|| f2_gzip_decompress(&gz))
    });
    group.finish();
}

criterion_group!(
    benches,
    bench_compress,
    bench_decompress,
    bench_compression_ratio,
    bench_gzip_inflate_only,
    bench_crc32_raw
);
criterion_main!(benches);
