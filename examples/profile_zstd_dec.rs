//! Decode-only profile harness for callgrind. Pre-computes C-compressed
//! frames for every BENCH_SUBSET file, then measures ours-decompress median
//! so the callgrind profile is dominated by the decoder hot path.
use std::io::{Cursor, Read};
use std::time::Instant;

fn load_bz2(path: &str) -> Vec<u8> {
    let bz2 = std::fs::read(path).expect("read corpus");
    let mut dec = bzip2::read::BzDecoder::new(&bz2[..]);
    let mut out = Vec::new();
    dec.read_to_end(&mut out).unwrap();
    out
}

fn main() {
    // A representative file — dickens is the largest text file in the subset.
    let input = load_bz2("benches/data/dickens.bz2");
    eprintln!("Loaded {} bytes from dickens", input.len());

    // Produce a C-zstd-compressed payload once so we only profile decode.
    let compressed = {
        let mut enc = zstd::stream::read::Encoder::new(&input[..], 3).unwrap();
        let mut out = Vec::new();
        enc.read_to_end(&mut out).unwrap();
        out
    };
    eprintln!("Compressed: {} bytes", compressed.len());

    eprintln!("\n--- ours zstd decompress (50 iters) ---");
    let mut times = Vec::new();
    for _ in 0..50 {
        let mut out = Vec::with_capacity(input.len());
        let t = Instant::now();
        libcramjam::zstd::decompress(&mut Cursor::new(&compressed), &mut out).unwrap();
        times.push(t.elapsed());
        std::hint::black_box(out);
    }
    times.sort();
    let median = times[times.len() / 2];
    let throughput = input.len() as f64 / 1024.0 / 1024.0 / median.as_secs_f64();
    eprintln!("  median: {:?}  ({:.0} MB/s)", median, throughput);
}
