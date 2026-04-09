//! Benchmarks comparing our pure-Rust XZ / LZMA against the C-backed `xz2` crate.
//!
//! Run with: cargo bench --bench xz_bench

use criterion::{
    criterion_group, criterion_main, BenchmarkId, Criterion, Throughput,
};
use std::io::{Cursor, Read, Write};

// ---------------------------------------------------------------------------
// Corpus helpers
// ---------------------------------------------------------------------------

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
// Wrappers
// ---------------------------------------------------------------------------

fn ours_compress(data: &[u8], level: u32) -> Vec<u8> {
    let mut out = Vec::with_capacity(data.len() / 2);
    libcramjam::xz::compress(
        &mut Cursor::new(data),
        &mut out,
        Some(level),
        None::<libcramjam::xz::Format>,
        None::<libcramjam::xz::Check>,
        None::<libcramjam::xz::Filters>,
        None::<libcramjam::xz::LzmaOptions>,
    )
    .unwrap();
    out
}

fn ours_decompress(data: &[u8]) -> Vec<u8> {
    let mut out = Vec::with_capacity(data.len() * 4);
    libcramjam::xz::decompress(&mut Cursor::new(data), &mut out).unwrap();
    out
}

fn c_compress(data: &[u8], level: u32) -> Vec<u8> {
    let mut enc = xz2::write::XzEncoder::new(Vec::new(), level);
    enc.write_all(data).unwrap();
    enc.finish().unwrap()
}

fn c_decompress(data: &[u8]) -> Vec<u8> {
    let mut dec = xz2::read::XzDecoder::new(data);
    let mut out = Vec::new();
    dec.read_to_end(&mut out).unwrap();
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
    let levels = [1u32, 6, 9];

    for (corpus_name, data) in &corpora {
        let mut group = c.benchmark_group(format!("xz_compress/{corpus_name}"));
        group.throughput(Throughput::Bytes(data.len() as u64));

        for &level in &levels {
            group.bench_with_input(BenchmarkId::new("ours", level), &level, |b, &l| {
                b.iter(|| ours_compress(data, l))
            });
            group.bench_with_input(BenchmarkId::new("c_xz", level), &level, |b, &l| {
                b.iter(|| c_compress(data, l))
            });
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

    for (corpus_name, data) in &corpora {
        let mut group = c.benchmark_group(format!("xz_decompress/{corpus_name}"));
        group.throughput(Throughput::Bytes(data.len() as u64));

        // Decode bytes produced by C xz at level 6 — both implementations
        // see the same input.
        let c_data = c_compress(data, 6);

        group.bench_function("ours", |b| b.iter(|| ours_decompress(&c_data)));
        group.bench_function("c_xz", |b| b.iter(|| c_decompress(&c_data)));

        group.finish();
    }
}

/// Compression ratio sanity check.
fn bench_compression_ratio(c: &mut Criterion) {
    let corpora: Vec<(&str, Vec<u8>)> = vec![
        ("text_100k", gen_text(100_000)),
        ("random_100k", gen_random(100_000)),
    ];

    let mut group = c.benchmark_group("xz_compression_ratio");
    group.sample_size(10);

    for (corpus_name, data) in &corpora {
        for level in [1u32, 6, 9] {
            let ours = ours_compress(data, level);
            let c_out = c_compress(data, level);
            let ratio_ours = ours.len() as f64 / data.len() as f64;
            let ratio_c = c_out.len() as f64 / data.len() as f64;
            eprintln!(
                "[ratio] {corpus_name} L{level}: ours={} ({:.2}%) c_xz={} ({:.2}%) delta={:+.2}%",
                ours.len(),
                ratio_ours * 100.0,
                c_out.len(),
                ratio_c * 100.0,
                (ratio_ours - ratio_c) * 100.0,
            );
            group.bench_function(format!("{corpus_name}/L{level}/ours"), |b| {
                b.iter(|| ours_compress(data, level))
            });
            group.bench_function(format!("{corpus_name}/L{level}/c_xz"), |b| {
                b.iter(|| c_compress(data, level))
            });
        }
    }
    group.finish();
}

criterion_group!(benches, bench_compress, bench_decompress, bench_compression_ratio);
criterion_main!(benches);
