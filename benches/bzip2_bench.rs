//! Benchmarks comparing our pure-Rust bzip2 against the C-backed `bzip2` crate.
//!
//! Run with: cargo bench --bench bzip2_bench
//!
//! Inputs come from the shared corpus under `benches/data/` (see `common.rs`).

use criterion::{
    criterion_group, criterion_main, BenchmarkId, Criterion, Throughput,
};
use std::io::{Cursor, Read, Write};

#[path = "common.rs"]
mod common;

// ---------------------------------------------------------------------------
// Wrappers
// ---------------------------------------------------------------------------

fn ours_compress(data: &[u8], level: u32) -> Vec<u8> {
    let mut out = Vec::with_capacity(data.len() / 2);
    libcramjam::bzip2::compress(&mut Cursor::new(data), &mut out, Some(level)).unwrap();
    out
}

fn ours_decompress(data: &[u8]) -> Vec<u8> {
    let mut out = Vec::with_capacity(data.len() * 4);
    libcramjam::bzip2::decompress(&mut Cursor::new(data), &mut out).unwrap();
    out
}

fn c_compress(data: &[u8], level: u32) -> Vec<u8> {
    let mut enc = bzip2::write::BzEncoder::new(Vec::new(), bzip2::Compression::new(level));
    enc.write_all(data).unwrap();
    enc.finish().unwrap()
}

fn c_decompress(data: &[u8]) -> Vec<u8> {
    let mut dec = bzip2::read::BzDecoder::new(data);
    let mut out = Vec::new();
    dec.read_to_end(&mut out).unwrap();
    out
}

const LEVELS: &[u32] = &[1, 6, 9];

// =========================================================================
// Benchmark groups
// =========================================================================

fn bench_compress(c: &mut Criterion) {
    for (corpus_name, data) in common::load_bench_subset() {
        let mut group = c.benchmark_group(format!("bzip2_compress/{corpus_name}"));
        group.throughput(Throughput::Bytes(data.len() as u64));
        group.sample_size(15);

        for &level in LEVELS {
            group.bench_with_input(BenchmarkId::new("ours", level), &level, |b, &l| {
                b.iter(|| ours_compress(data, l))
            });
            group.bench_with_input(BenchmarkId::new("c_bzip2", level), &level, |b, &l| {
                b.iter(|| c_compress(data, l))
            });
        }
        group.finish();
    }
}

fn bench_decompress(c: &mut Criterion) {
    for (corpus_name, data) in common::load_bench_subset() {
        let mut group = c.benchmark_group(format!("bzip2_decompress/{corpus_name}"));
        group.throughput(Throughput::Bytes(data.len() as u64));
        group.sample_size(15);

        // Decode bytes produced by C bzip2 at level 6 — both implementations
        // see the same input.
        let c_data = c_compress(data, 6);

        group.bench_function("ours", |b| b.iter(|| ours_decompress(&c_data)));
        group.bench_function("c_bzip2", |b| b.iter(|| c_decompress(&c_data)));

        group.finish();
    }
}

/// Compression ratio sanity check.
fn bench_compression_ratio(c: &mut Criterion) {
    let mut group = c.benchmark_group("bzip2_compression_ratio");
    group.sample_size(10);

    for (corpus_name, data) in common::load_bench_subset() {
        for &level in LEVELS {
            let ours = ours_compress(data, level);
            let c_out = c_compress(data, level);
            let ratio_ours = ours.len() as f64 / data.len() as f64;
            let ratio_c = c_out.len() as f64 / data.len() as f64;
            eprintln!(
                "[ratio] {corpus_name} L{level}: ours={} ({:.1}%) c_bzip2={} ({:.1}%) delta={:+.2}pp",
                ours.len(),
                ratio_ours * 100.0,
                c_out.len(),
                ratio_c * 100.0,
                (ratio_ours - ratio_c) * 100.0,
            );
            group.bench_function(format!("{corpus_name}/L{level}/ours"), |b| {
                b.iter(|| ours_compress(data, level))
            });
            group.bench_function(format!("{corpus_name}/L{level}/c_bzip2"), |b| {
                b.iter(|| c_compress(data, level))
            });
        }
    }
    group.finish();
}

criterion_group!(benches, bench_compress, bench_decompress, bench_compression_ratio);
criterion_main!(benches);
