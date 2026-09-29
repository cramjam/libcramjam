//! Decode-only lz4 profile harness (for callgrind) with an interleaved
//! ours-vs-C timing so the ratio is meaningful even on a loaded machine.
//! `ITERS` env var overrides the iteration count (default 100).
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
    let input = load_bz2("benches/data/dickens.bz2");
    eprintln!("Loaded {} bytes from dickens", input.len());

    // Produce a C-lz4-compressed payload once so we only profile decode.
    let compressed = {
        let mut buf = Vec::new();
        let mut enc = lz4::EncoderBuilder::new().level(4).build(&mut buf).unwrap();
        std::io::copy(&mut &input[..], &mut enc).unwrap();
        let (_, r) = enc.finish();
        r.unwrap();
        buf
    };
    eprintln!("Compressed: {} bytes", compressed.len());

    let iters: usize = std::env::var("ITERS").ok().and_then(|s| s.parse().ok()).unwrap_or(100);
    let time_c = std::env::var("TIME_C").is_ok();
    eprintln!("\n--- ours lz4 decompress ({iters} iters{}) ---", if time_c { ", interleaved with C" } else { "" });
    let mut ours = Vec::new();
    let mut theirs = Vec::new();
    for _ in 0..iters {
        let mut out = Vec::with_capacity(input.len());
        let t = Instant::now();
        libcramjam::lz4::decompress(&mut Cursor::new(&compressed), &mut out).unwrap();
        ours.push(t.elapsed());
        assert_eq!(out.len(), input.len());
        std::hint::black_box(out);
        if time_c {
            let mut out = Vec::with_capacity(input.len());
            let t = Instant::now();
            lz4::Decoder::new(&compressed[..]).unwrap().read_to_end(&mut out).unwrap();
            theirs.push(t.elapsed());
            std::hint::black_box(out);
        }
    }
    ours.sort();
    let median = ours[ours.len() / 2];
    let throughput = input.len() as f64 / 1024.0 / 1024.0 / median.as_secs_f64();
    eprintln!("  ours median: {:?}  ({:.0} MB/s)", median, throughput);
    if time_c {
        theirs.sort();
        let cm = theirs[theirs.len() / 2];
        eprintln!("  C    median: {:?}  ({:.0} MB/s)  ours/C = {:.2}x", cm,
            input.len() as f64 / 1024.0 / 1024.0 / cm.as_secs_f64(),
            median.as_secs_f64() / cm.as_secs_f64());
    }
}
