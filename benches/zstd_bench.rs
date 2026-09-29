//! Benchmarks comparing our pure-Rust zstd against the C-backed `zstd` crate.
//!
//! Run with: cargo bench --bench zstd_bench
//!
//! Inputs are loaded from the shared benchmark corpus under `benches/data/`
//! (see `benches/common.rs`).  Files are stored as `.bz2` to keep the repo
//! small and decompressed once on first use.

use criterion::{
    criterion_group, criterion_main, BenchmarkId, Criterion, Throughput,
};
use std::io::{Cursor, Read};

#[path = "common.rs"]
mod common;

// ---------------------------------------------------------------------------
// Wrapper helpers — keep closure bodies tiny for criterion
// ---------------------------------------------------------------------------

fn ours_compress(data: &[u8], level: i32) -> Vec<u8> {
    let mut out = Vec::with_capacity(data.len());
    libcramjam::zstd::compress(&mut Cursor::new(data), &mut out, Some(level), Some(data.len())).unwrap();
    out
}

fn ours_decompress(data: &[u8]) -> Vec<u8> {
    let mut out = Vec::with_capacity(data.len() * 4);
    libcramjam::zstd::decompress(&mut Cursor::new(data), &mut out).unwrap();
    out
}

fn c_compress(data: &[u8], level: i32) -> Vec<u8> {
    let mut enc = zstd::stream::read::Encoder::new(data, level).unwrap();
    let mut out = Vec::new();
    enc.read_to_end(&mut out).unwrap();
    out
}

fn c_decompress(data: &[u8]) -> Vec<u8> {
    let mut dec = zstd::stream::read::Decoder::new(data).unwrap();
    let mut out = Vec::new();
    dec.read_to_end(&mut out).unwrap();
    out
}

const LEVELS: &[i32] = &[1, 3, 6, 9];

// =========================================================================
// Benchmark groups
// =========================================================================

fn bench_compress(c: &mut Criterion) {
    for (corpus_name, data) in common::load_bench_subset() {
        let mut group = c.benchmark_group(format!("zstd_compress/{corpus_name}"));
        group.throughput(Throughput::Bytes(data.len() as u64));
        group.sample_size(20);

        for &level in LEVELS {
            group.bench_with_input(BenchmarkId::new("ours", level), &level, |b, &l| {
                b.iter(|| ours_compress(data, l))
            });
            group.bench_with_input(BenchmarkId::new("c_zstd", level), &level, |b, &l| {
                b.iter(|| c_compress(data, l))
            });
        }
        group.finish();
    }
}

fn bench_decompress(c: &mut Criterion) {
    // Bench decompression of data compressed by the C reference at level 6,
    // which exercises FSE+Huffman+sequences and is the common case for I/O.
    let level = 6i32;

    for (corpus_name, data) in common::load_bench_subset() {
        let mut group = c.benchmark_group(format!("zstd_decompress/{corpus_name}"));
        group.throughput(Throughput::Bytes(data.len() as u64));
        group.sample_size(20);

        // Pre-compress with C zstd at level 6.  Both implementations decode
        // the same bytes — apples-to-apples.
        let c_data = c_compress(data, level);

        group.bench_function("ours", |b| b.iter(|| ours_decompress(&c_data)));
        group.bench_function("c_zstd", |b| b.iter(|| c_decompress(&c_data)));

        group.finish();
    }
}

/// Compression ratio sanity check: prints a side-by-side ratio table for each
/// corpus / level pair.  Output goes to stderr; the criterion group itself
/// just makes sure the encoders run.
fn bench_compression_ratio(c: &mut Criterion) {
    let mut group = c.benchmark_group("zstd_compression_ratio");
    group.sample_size(10);

    for (corpus_name, data) in common::load_bench_subset() {
        for &level in LEVELS {
            let ours = ours_compress(data, level);
            let c_out = c_compress(data, level);
            let ratio_ours = ours.len() as f64 / data.len() as f64;
            let ratio_c = c_out.len() as f64 / data.len() as f64;
            eprintln!(
                "[ratio] {corpus_name} L{level}: ours={} ({:.1}%) c_zstd={} ({:.1}%) delta={:+.2}pp",
                ours.len(),
                ratio_ours * 100.0,
                c_out.len(),
                ratio_c * 100.0,
                (ratio_ours - ratio_c) * 100.0,
            );
            group.bench_function(
                format!("{corpus_name}/L{level}/ours"),
                |b| b.iter(|| ours_compress(data, level)),
            );
            group.bench_function(
                format!("{corpus_name}/L{level}/c_zstd"),
                |b| b.iter(|| c_compress(data, level)),
            );
        }
    }
    group.finish();
}

criterion_group!(benches, bench_compress, bench_decompress, bench_compression_ratio);
criterion_main!(benches);
