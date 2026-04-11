//! High-level codec summary benchmark.
//!
//! Runs every codec at representative levels across the BENCH_SUBSET corpus,
//! measures median compress / decompress / roundtrip times, and prints a
//! single reproducible ASCII table comparing our pure-Rust implementations
//! against the C reference libraries.
//!
//! Usage:
//!   cargo bench --bench summary
//!
//! The report goes to stdout; criterion is NOT used — this is a plain binary
//! so the output is self-contained and easy to paste into issues / docs.

#[path = "common.rs"]
mod common;

use std::io::{Cursor, Read};
use std::time::{Duration, Instant};

// ---------------------------------------------------------------------------
// Measurement helpers
// ---------------------------------------------------------------------------

const WARMUP: usize = 1;
const ITERS: usize = 5;

fn median(times: &mut [Duration]) -> Duration {
    times.sort();
    times[times.len() / 2]
}

fn bench_fn<F: FnMut()>(mut f: F) -> Duration {
    for _ in 0..WARMUP {
        f();
    }
    let mut times = Vec::with_capacity(ITERS);
    for _ in 0..ITERS {
        let t0 = Instant::now();
        f();
        times.push(t0.elapsed());
    }
    median(&mut times)
}

// ---------------------------------------------------------------------------
// Codec wrappers — ours
// ---------------------------------------------------------------------------

fn ours_deflate_compress(data: &[u8], level: u32) -> Vec<u8> {
    let mut out = Vec::with_capacity(data.len());
    libcramjam::deflate::compress(&mut Cursor::new(data), &mut out, Some(level)).unwrap();
    out
}
fn ours_deflate_decompress(data: &[u8]) -> Vec<u8> {
    let mut out = Vec::new();
    libcramjam::deflate::decompress(&mut Cursor::new(data), &mut out).unwrap();
    out
}

fn ours_gzip_compress(data: &[u8], level: u32) -> Vec<u8> {
    let mut out = Vec::with_capacity(data.len());
    libcramjam::gzip::compress(&mut Cursor::new(data), &mut out, Some(level)).unwrap();
    out
}
fn ours_gzip_decompress(data: &[u8]) -> Vec<u8> {
    let mut out = Vec::new();
    libcramjam::gzip::decompress(&mut Cursor::new(data), &mut out).unwrap();
    out
}

fn ours_zstd_compress(data: &[u8], level: i32) -> Vec<u8> {
    let mut out = Vec::with_capacity(data.len());
    libcramjam::zstd::compress(&mut Cursor::new(data), &mut out, Some(level), Some(data.len())).unwrap();
    out
}
fn ours_zstd_decompress(data: &[u8]) -> Vec<u8> {
    let mut out = Vec::new();
    libcramjam::zstd::decompress(&mut Cursor::new(data), &mut out).unwrap();
    out
}

fn ours_lz4_compress(data: &[u8], level: u32) -> Vec<u8> {
    let mut out = Vec::with_capacity(data.len());
    libcramjam::lz4::compress(&mut Cursor::new(data), &mut out, Some(level)).unwrap();
    out
}
fn ours_lz4_decompress(data: &[u8]) -> Vec<u8> {
    let mut out = Vec::new();
    libcramjam::lz4::decompress(&mut Cursor::new(data), &mut out).unwrap();
    out
}

fn ours_bzip2_compress(data: &[u8], level: u32) -> Vec<u8> {
    let mut out = Vec::with_capacity(data.len());
    libcramjam::bzip2::compress(&mut Cursor::new(data), &mut out, Some(level)).unwrap();
    out
}
fn ours_bzip2_decompress(data: &[u8]) -> Vec<u8> {
    let mut out = Vec::new();
    libcramjam::bzip2::decompress(&mut Cursor::new(data), &mut out).unwrap();
    out
}

fn ours_xz_compress(data: &[u8], level: u32) -> Vec<u8> {
    let mut out = Vec::with_capacity(data.len());
    libcramjam::xz::compress(
        &mut Cursor::new(data), &mut out, Some(level),
        None::<libcramjam::xz::Format>, None::<libcramjam::xz::Check>,
        None::<libcramjam::xz::Filters>, None::<libcramjam::xz::LzmaOptions>,
    ).unwrap();
    out
}
fn ours_xz_decompress(data: &[u8]) -> Vec<u8> {
    let mut out = Vec::new();
    libcramjam::xz::decompress(&mut Cursor::new(data), &mut out).unwrap();
    out
}

// ---------------------------------------------------------------------------
// Codec wrappers — C reference
// ---------------------------------------------------------------------------

fn c_deflate_compress(data: &[u8], level: u32) -> Vec<u8> {
    use flate2::write::DeflateEncoder;
    use std::io::Write;
    let mut enc = DeflateEncoder::new(Vec::new(), flate2::Compression::new(level));
    enc.write_all(data).unwrap();
    enc.finish().unwrap()
}
fn c_deflate_decompress(data: &[u8]) -> Vec<u8> {
    let mut dec = flate2::read::DeflateDecoder::new(data);
    let mut out = Vec::new();
    dec.read_to_end(&mut out).unwrap();
    out
}

fn c_gzip_compress(data: &[u8], level: u32) -> Vec<u8> {
    use flate2::write::GzEncoder;
    use std::io::Write;
    let mut enc = GzEncoder::new(Vec::new(), flate2::Compression::new(level));
    enc.write_all(data).unwrap();
    enc.finish().unwrap()
}
fn c_gzip_decompress(data: &[u8]) -> Vec<u8> {
    let mut dec = flate2::read::GzDecoder::new(data);
    let mut out = Vec::new();
    dec.read_to_end(&mut out).unwrap();
    out
}

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

fn c_lz4_compress(data: &[u8], level: u32) -> Vec<u8> {
    use std::io::Write;
    let mut enc = lz4::EncoderBuilder::new()
        .level(level)
        .auto_flush(true)
        .build(Vec::new())
        .unwrap();
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

fn c_bzip2_compress(data: &[u8], level: u32) -> Vec<u8> {
    use bzip2::write::BzEncoder;
    use std::io::Write;
    let mut enc = BzEncoder::new(Vec::new(), bzip2::Compression::new(level));
    enc.write_all(data).unwrap();
    enc.finish().unwrap()
}
fn c_bzip2_decompress(data: &[u8]) -> Vec<u8> {
    let mut dec = bzip2::read::BzDecoder::new(data);
    let mut out = Vec::new();
    dec.read_to_end(&mut out).unwrap();
    out
}

fn c_xz_compress(data: &[u8], level: u32) -> Vec<u8> {
    use std::io::Write;
    let mut enc = xz2::write::XzEncoder::new(Vec::new(), level);
    enc.write_all(data).unwrap();
    enc.finish().unwrap()
}
fn c_xz_decompress(data: &[u8]) -> Vec<u8> {
    let mut dec = xz2::read::XzDecoder::new(data);
    let mut out = Vec::new();
    dec.read_to_end(&mut out).unwrap();
    out
}

// ---------------------------------------------------------------------------
// Report
// ---------------------------------------------------------------------------

struct Row {
    codec: &'static str,
    level: String,
    input_mb: f64,
    ours_compress: Duration,
    ours_decompress: Duration,
    ours_ratio: f64,
    c_compress: Duration,
    c_decompress: Duration,
    c_ratio: f64,
}

fn fmt_dur(d: Duration) -> String {
    let ms = d.as_secs_f64() * 1000.0;
    if ms >= 1000.0 {
        format!("{:.2}s", ms / 1000.0)
    } else {
        format!("{:.1}ms", ms)
    }
}

fn fmt_speed(bytes: f64, d: Duration) -> String {
    let mb_s = bytes / (1024.0 * 1024.0) / d.as_secs_f64();
    if mb_s >= 1000.0 {
        format!("{:.1} GB/s", mb_s / 1024.0)
    } else {
        format!("{:.0} MB/s", mb_s)
    }
}

fn fmt_ratio_cmp(ours: f64, theirs: f64) -> String {
    let x = ours / theirs;
    if x >= 1.0 {
        format!("{:.2}x slower", x)
    } else {
        format!("{:.2}x faster", 1.0 / x)
    }
}

fn print_report(rows: &[Row]) {
    let input_bytes = rows[0].input_mb * 1024.0 * 1024.0;

    println!();
    println!("libcramjam codec summary");
    println!("========================");
    println!("Corpus: BENCH_SUBSET ({} files, {:.1} MB)", common::BENCH_SUBSET.len(), rows[0].input_mb);
    println!("Iterations: {} (median of {})", WARMUP + ITERS, ITERS);

    // ==================================================================
    // Section 1: Detailed per-codec per-level table
    // ==================================================================
    println!();
    println!("1. Detailed results");
    println!("-------------------");
    println!();
    println!(
        "{:<10} {:>5}  {:>10} {:>10} {:>10}  {:>7}  {:>10} {:>10} {:>10}  {:>7}  {:>7}",
        "codec", "level",
        "comp", "decomp", "round",
        "ratio",
        "C comp", "C decomp", "C round",
        "C ratio",
        "gap",
    );
    println!("{}", "-".repeat(118));

    for r in rows {
        let ours_round = r.ours_compress + r.ours_decompress;
        let c_round = r.c_compress + r.c_decompress;
        let gap = r.ours_ratio - r.c_ratio;
        println!(
            "{:<10} {:>5}  {:>10} {:>10} {:>10}  {:>6.1}%  {:>10} {:>10} {:>10}  {:>6.1}%  {:>+6.1}pp",
            r.codec, r.level,
            fmt_dur(r.ours_compress), fmt_dur(r.ours_decompress), fmt_dur(ours_round),
            r.ours_ratio * 100.0,
            fmt_dur(r.c_compress), fmt_dur(r.c_decompress), fmt_dur(c_round),
            r.c_ratio * 100.0,
            gap * 100.0,
        );
    }

    // ==================================================================
    // Section 2: Median timing comparison (at default levels)
    // ==================================================================
    println!();
    println!("2. Median timing at default levels");
    println!("----------------------------------");
    println!();
    println!(
        "  {:<10}  {:>10} {:>10} {:>10}  {:>10} {:>10} {:>10}  {:>14} {:>14}",
        "codec",
        "ours comp", "ours dec", "ours rnd",
        "C comp", "C dec", "C rnd",
        "comp vs C", "dec vs C",
    );
    println!("  {}", "-".repeat(128));
    for r in rows {
        if !is_default_level(r) { continue; }
        let ours_round = r.ours_compress + r.ours_decompress;
        let c_round = r.c_compress + r.c_decompress;
        println!(
            "  {:<10}  {:>10} {:>10} {:>10}  {:>10} {:>10} {:>10}  {:>14} {:>14}",
            r.codec,
            fmt_dur(r.ours_compress), fmt_dur(r.ours_decompress), fmt_dur(ours_round),
            fmt_dur(r.c_compress), fmt_dur(r.c_decompress), fmt_dur(c_round),
            fmt_ratio_cmp(r.ours_compress.as_secs_f64(), r.c_compress.as_secs_f64()),
            fmt_ratio_cmp(r.ours_decompress.as_secs_f64(), r.c_decompress.as_secs_f64()),
        );
    }

    // ==================================================================
    // Section 3: Throughput comparison (at default levels)
    // ==================================================================
    println!();
    println!("3. Throughput at default levels");
    println!("------------------------------");
    println!();
    println!(
        "  {:<10}  {:>12} {:>12}  {:>12} {:>12}",
        "codec",
        "ours comp", "ours decomp",
        "C comp", "C decomp",
    );
    println!("  {}", "-".repeat(60));
    for r in rows {
        if !is_default_level(r) { continue; }
        println!(
            "  {:<10}  {:>12} {:>12}  {:>12} {:>12}",
            r.codec,
            fmt_speed(input_bytes, r.ours_compress),
            fmt_speed(input_bytes, r.ours_decompress),
            fmt_speed(input_bytes, r.c_compress),
            fmt_speed(input_bytes, r.c_decompress),
        );
    }

    // ==================================================================
    // Section 4: Compression ratio comparison (at default levels)
    // ==================================================================
    println!();
    println!("4. Compression ratio at default levels");
    println!("--------------------------------------");
    println!();
    println!(
        "  {:<10}  {:>10} {:>10}  {:>8}  {:>44}",
        "codec", "ours", "C ref", "gap", "",
    );
    println!("  {}", "-".repeat(90));
    for r in rows {
        if !is_default_level(r) { continue; }
        let gap_pp = (r.ours_ratio - r.c_ratio) * 100.0;
        let bar_len = 40;
        let ours_bar = (r.ours_ratio * bar_len as f64) as usize;
        let c_bar = (r.c_ratio * bar_len as f64) as usize;
        let bar: String = (0..bar_len)
            .map(|i| if i < ours_bar.min(c_bar) { '#' } else if i < ours_bar { '>' } else if i < c_bar { '<' } else { ' ' })
            .collect();
        println!(
            "  {:<10}  {:>9.1}% {:>9.1}%  {:>+6.1}pp   [{}] # ours  < C better",
            r.codec,
            r.ours_ratio * 100.0,
            r.c_ratio * 100.0,
            gap_pp,
            bar,
        );
    }

    // ==================================================================
    // Section 5: Codec ranking by roundtrip speed (default levels)
    // ==================================================================
    println!();
    println!("5. Codec ranking by roundtrip speed (default level, fastest first)");
    println!("------------------------------------------------------------------");
    println!();
    let mut defaults: Vec<&Row> = rows.iter().filter(|r| is_default_level(r)).collect();
    defaults.sort_by(|a, b| {
        let a_rt = a.ours_compress + a.ours_decompress;
        let b_rt = b.ours_compress + b.ours_decompress;
        a_rt.cmp(&b_rt)
    });
    println!(
        "  {:>4}  {:<10}  {:>10}  {:>12} {:>12}  {:>7}",
        "#", "codec", "roundtrip", "comp tput", "dec tput", "ratio",
    );
    println!("  {}", "-".repeat(66));
    for (i, r) in defaults.iter().enumerate() {
        let rt = r.ours_compress + r.ours_decompress;
        println!(
            "  {:>4}  {:<10}  {:>10}  {:>12} {:>12}  {:>6.1}%",
            i + 1,
            r.codec,
            fmt_dur(rt),
            fmt_speed(input_bytes, r.ours_compress),
            fmt_speed(input_bytes, r.ours_decompress),
            r.ours_ratio * 100.0,
        );
    }

    println!();
}

fn is_default_level(r: &Row) -> bool {
    match r.codec {
        "deflate" | "gzip" | "bzip2" | "xz" => r.level == "6",
        "zstd" => r.level == "3",
        "lz4" => r.level == "4",
        _ => false,
    }
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------

fn main() {
    // Load and concatenate corpus for aggregate timing.
    let corpus: Vec<(&str, &[u8])> = common::load_bench_subset();
    let total_bytes: usize = corpus.iter().map(|(_, d)| d.len()).sum();
    let total_mb = total_bytes as f64 / (1024.0 * 1024.0);

    eprintln!(
        "Loaded {} files, {:.1} MB total. Running benchmarks...",
        corpus.len(),
        total_mb
    );

    let mut rows: Vec<Row> = Vec::new();

    // --- Deflate ---
    for &level in &[1u32, 6, 9] {
        eprint!("  deflate L{level}...");
        let ours_c = bench_fn(|| { for (_, d) in &corpus { let _ = ours_deflate_compress(d, level); } });
        let compressed: Vec<Vec<u8>> = corpus.iter().map(|(_, d)| ours_deflate_compress(d, level)).collect();
        let ours_d = bench_fn(|| { for c in &compressed { let _ = ours_deflate_decompress(c); } });
        let ours_size: usize = compressed.iter().map(|c| c.len()).sum();
        let c_c = bench_fn(|| { for (_, d) in &corpus { let _ = c_deflate_compress(d, level); } });
        let c_compressed: Vec<Vec<u8>> = corpus.iter().map(|(_, d)| c_deflate_compress(d, level)).collect();
        let c_d = bench_fn(|| { for c in &c_compressed { let _ = c_deflate_decompress(c); } });
        let c_size: usize = c_compressed.iter().map(|c| c.len()).sum();
        eprintln!(" done");
        rows.push(Row {
            codec: "deflate", level: format!("{level}"), input_mb: total_mb,
            ours_compress: ours_c, ours_decompress: ours_d, ours_ratio: ours_size as f64 / total_bytes as f64,
            c_compress: c_c, c_decompress: c_d, c_ratio: c_size as f64 / total_bytes as f64,
        });
    }

    // --- Gzip ---
    for &level in &[1u32, 6, 9] {
        eprint!("  gzip L{level}...");
        let ours_c = bench_fn(|| { for (_, d) in &corpus { let _ = ours_gzip_compress(d, level); } });
        let compressed: Vec<Vec<u8>> = corpus.iter().map(|(_, d)| ours_gzip_compress(d, level)).collect();
        let ours_d = bench_fn(|| { for c in &compressed { let _ = ours_gzip_decompress(c); } });
        let ours_size: usize = compressed.iter().map(|c| c.len()).sum();
        let c_c = bench_fn(|| { for (_, d) in &corpus { let _ = c_gzip_compress(d, level); } });
        let c_compressed: Vec<Vec<u8>> = corpus.iter().map(|(_, d)| c_gzip_compress(d, level)).collect();
        let c_d = bench_fn(|| { for c in &c_compressed { let _ = c_gzip_decompress(c); } });
        let c_size: usize = c_compressed.iter().map(|c| c.len()).sum();
        eprintln!(" done");
        rows.push(Row {
            codec: "gzip", level: format!("{level}"), input_mb: total_mb,
            ours_compress: ours_c, ours_decompress: ours_d, ours_ratio: ours_size as f64 / total_bytes as f64,
            c_compress: c_c, c_decompress: c_d, c_ratio: c_size as f64 / total_bytes as f64,
        });
    }

    // --- Zstd ---
    for &level in &[1i32, 3, 6, 9] {
        eprint!("  zstd L{level}...");
        let ours_c = bench_fn(|| { for (_, d) in &corpus { let _ = ours_zstd_compress(d, level); } });
        let compressed: Vec<Vec<u8>> = corpus.iter().map(|(_, d)| ours_zstd_compress(d, level)).collect();
        let ours_d = bench_fn(|| { for c in &compressed { let _ = ours_zstd_decompress(c); } });
        let ours_size: usize = compressed.iter().map(|c| c.len()).sum();
        let c_c = bench_fn(|| { for (_, d) in &corpus { let _ = c_zstd_compress(d, level); } });
        let c_compressed: Vec<Vec<u8>> = corpus.iter().map(|(_, d)| c_zstd_compress(d, level)).collect();
        let c_d = bench_fn(|| { for c in &c_compressed { let _ = c_zstd_decompress(c); } });
        let c_size: usize = c_compressed.iter().map(|c| c.len()).sum();
        eprintln!(" done");
        rows.push(Row {
            codec: "zstd", level: format!("{level}"), input_mb: total_mb,
            ours_compress: ours_c, ours_decompress: ours_d, ours_ratio: ours_size as f64 / total_bytes as f64,
            c_compress: c_c, c_decompress: c_d, c_ratio: c_size as f64 / total_bytes as f64,
        });
    }

    // --- LZ4 ---
    for &level in &[1u32, 4, 9] {
        eprint!("  lz4 L{level}...");
        let ours_c = bench_fn(|| { for (_, d) in &corpus { let _ = ours_lz4_compress(d, level); } });
        let compressed: Vec<Vec<u8>> = corpus.iter().map(|(_, d)| ours_lz4_compress(d, level)).collect();
        let ours_d = bench_fn(|| { for c in &compressed { let _ = ours_lz4_decompress(c); } });
        let ours_size: usize = compressed.iter().map(|c| c.len()).sum();
        let c_c = bench_fn(|| { for (_, d) in &corpus { let _ = c_lz4_compress(d, level); } });
        let c_compressed: Vec<Vec<u8>> = corpus.iter().map(|(_, d)| c_lz4_compress(d, level)).collect();
        let c_d = bench_fn(|| { for c in &c_compressed { let _ = c_lz4_decompress(c); } });
        let c_size: usize = c_compressed.iter().map(|c| c.len()).sum();
        eprintln!(" done");
        rows.push(Row {
            codec: "lz4", level: format!("{level}"), input_mb: total_mb,
            ours_compress: ours_c, ours_decompress: ours_d, ours_ratio: ours_size as f64 / total_bytes as f64,
            c_compress: c_c, c_decompress: c_d, c_ratio: c_size as f64 / total_bytes as f64,
        });
    }

    // --- Bzip2 ---
    for &level in &[1u32, 6, 9] {
        eprint!("  bzip2 L{level}...");
        let ours_c = bench_fn(|| { for (_, d) in &corpus { let _ = ours_bzip2_compress(d, level); } });
        let compressed: Vec<Vec<u8>> = corpus.iter().map(|(_, d)| ours_bzip2_compress(d, level)).collect();
        let ours_d = bench_fn(|| { for c in &compressed { let _ = ours_bzip2_decompress(c); } });
        let ours_size: usize = compressed.iter().map(|c| c.len()).sum();
        let c_c = bench_fn(|| { for (_, d) in &corpus { let _ = c_bzip2_compress(d, level); } });
        let c_compressed: Vec<Vec<u8>> = corpus.iter().map(|(_, d)| c_bzip2_compress(d, level)).collect();
        let c_d = bench_fn(|| { for c in &c_compressed { let _ = c_bzip2_decompress(c); } });
        let c_size: usize = c_compressed.iter().map(|c| c.len()).sum();
        eprintln!(" done");
        rows.push(Row {
            codec: "bzip2", level: format!("{level}"), input_mb: total_mb,
            ours_compress: ours_c, ours_decompress: ours_d, ours_ratio: ours_size as f64 / total_bytes as f64,
            c_compress: c_c, c_decompress: c_d, c_ratio: c_size as f64 / total_bytes as f64,
        });
    }

    // --- XZ ---
    for &level in &[1u32, 6] {
        eprint!("  xz L{level}...");
        let ours_c = bench_fn(|| { for (_, d) in &corpus { let _ = ours_xz_compress(d, level); } });
        let compressed: Vec<Vec<u8>> = corpus.iter().map(|(_, d)| ours_xz_compress(d, level)).collect();
        let ours_d = bench_fn(|| { for c in &compressed { let _ = ours_xz_decompress(c); } });
        let ours_size: usize = compressed.iter().map(|c| c.len()).sum();
        let c_c = bench_fn(|| { for (_, d) in &corpus { let _ = c_xz_compress(d, level); } });
        let c_compressed: Vec<Vec<u8>> = corpus.iter().map(|(_, d)| c_xz_compress(d, level)).collect();
        let c_d = bench_fn(|| { for c in &c_compressed { let _ = c_xz_decompress(c); } });
        let c_size: usize = c_compressed.iter().map(|c| c.len()).sum();
        eprintln!(" done");
        rows.push(Row {
            codec: "xz", level: format!("{level}"), input_mb: total_mb,
            ours_compress: ours_c, ours_decompress: ours_d, ours_ratio: ours_size as f64 / total_bytes as f64,
            c_compress: c_c, c_decompress: c_d, c_ratio: c_size as f64 / total_bytes as f64,
        });
    }

    print_report(&rows);
}
