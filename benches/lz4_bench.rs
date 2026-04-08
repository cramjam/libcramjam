//! Benchmarks comparing our pure-Rust lz4 against the C-backed `lz4` crate.
//!
//! Run with: cargo bench --bench lz4_bench

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
// Wrappers — keep closure bodies tiny for criterion
// ---------------------------------------------------------------------------

fn ours_compress(data: &[u8]) -> Vec<u8> {
    let mut out = Vec::with_capacity(data.len());
    libcramjam::lz4::compress(&mut Cursor::new(data), &mut out, None).unwrap();
    out
}

fn ours_decompress(data: &[u8]) -> Vec<u8> {
    let mut out = Vec::with_capacity(data.len() * 4);
    libcramjam::lz4::decompress(&mut Cursor::new(data), &mut out).unwrap();
    out
}

fn c_compress(data: &[u8]) -> Vec<u8> {
    let mut enc = lz4::EncoderBuilder::new()
        .level(4)
        .auto_flush(true)
        .build(Vec::new())
        .unwrap();
    enc.write_all(data).unwrap();
    let (out, r) = enc.finish();
    r.unwrap();
    out
}

fn c_decompress(data: &[u8]) -> Vec<u8> {
    let mut dec = lz4::Decoder::new(data).unwrap();
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

    for (corpus_name, data) in &corpora {
        let mut group = c.benchmark_group(format!("lz4_compress/{corpus_name}"));
        group.throughput(Throughput::Bytes(data.len() as u64));

        group.bench_function(BenchmarkId::new("ours", "default"), |b| {
            b.iter(|| ours_compress(data))
        });
        group.bench_function(BenchmarkId::new("c_lz4", "default"), |b| {
            b.iter(|| c_compress(data))
        });
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
        let mut group = c.benchmark_group(format!("lz4_decompress/{corpus_name}"));
        group.throughput(Throughput::Bytes(data.len() as u64));

        // Apples-to-apples: both implementations decode the same C-produced bytes.
        let c_data = c_compress(data);

        group.bench_function("ours", |b| b.iter(|| ours_decompress(&c_data)));
        group.bench_function("c_lz4", |b| b.iter(|| c_decompress(&c_data)));

        group.finish();
    }
}

/// Compression ratio sanity check.
fn bench_compression_ratio(c: &mut Criterion) {
    let corpora: Vec<(&str, Vec<u8>)> = vec![
        ("text_100k", gen_text(100_000)),
        ("random_100k", gen_random(100_000)),
    ];

    let mut group = c.benchmark_group("lz4_compression_ratio");
    group.sample_size(10);

    for (corpus_name, data) in &corpora {
        let ours = ours_compress(data);
        let c_out = c_compress(data);
        let ratio_ours = ours.len() as f64 / data.len() as f64;
        let ratio_c = c_out.len() as f64 / data.len() as f64;
        eprintln!(
            "[ratio] {corpus_name}: ours={} ({:.1}%) c_lz4={} ({:.1}%) delta={:+.1}%",
            ours.len(),
            ratio_ours * 100.0,
            c_out.len(),
            ratio_c * 100.0,
            (ratio_ours - ratio_c) * 100.0,
        );
        group.bench_function(format!("{corpus_name}/ours"), |b| {
            b.iter(|| ours_compress(data))
        });
        group.bench_function(format!("{corpus_name}/c_lz4"), |b| {
            b.iter(|| c_compress(data))
        });
    }
    group.finish();
}

criterion_group!(benches, bench_compress, bench_decompress, bench_compression_ratio);
criterion_main!(benches);
