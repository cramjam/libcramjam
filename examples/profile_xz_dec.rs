//! Decode-only profile harness for xz (callgrind). Compresses dickens with C
//! xz once, then times our decoder. ITERS env var overrides the count.
use std::io::{Cursor, Read};
use std::time::Instant;

fn main() {
    let bz2 = std::fs::read("benches/data/dickens.bz2").expect("read corpus");
    let mut input = Vec::new();
    bzip2::read::BzDecoder::new(&bz2[..]).read_to_end(&mut input).unwrap();
    let mut compressed = Vec::new();
    xz2::read::XzEncoder::new(&input[..], 6).read_to_end(&mut compressed).unwrap();
    eprintln!("input {} bytes, xz {} bytes", input.len(), compressed.len());

    let iters: usize = std::env::var("ITERS").ok().and_then(|s| s.parse().ok()).unwrap_or(10);
    let mut times = Vec::new();
    for _ in 0..iters {
        let mut out = Vec::with_capacity(input.len());
        let t = Instant::now();
        libcramjam::xz::decompress(&mut Cursor::new(&compressed), &mut out).unwrap();
        times.push(t.elapsed());
        assert_eq!(out.len(), input.len());
        std::hint::black_box(out);
    }
    times.sort();
    let median = times[times.len() / 2];
    eprintln!("median: {:?} ({:.0} MB/s)", median, input.len() as f64 / 1048576.0 / median.as_secs_f64());
}
