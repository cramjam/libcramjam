//! Manual time-instrumented profile of bzip2 compress.
use std::io::{Cursor, Read};
use std::time::Instant;

fn main() {
    let path = std::path::Path::new("benches/data/dickens.bz2");
    let bz2 = std::fs::read(path).expect("read corpus");
    let mut dec = bzip2::read::BzDecoder::new(&bz2[..]);
    let mut input = Vec::new();
    dec.read_to_end(&mut input).unwrap();
    eprintln!("Loaded {} bytes from dickens", input.len());

    eprintln!("\n--- ours bzip2 compress L6 ---");
    let mut times = Vec::new();
    for _ in 0..5 {
        let mut out = Vec::with_capacity(input.len());
        let t = Instant::now();
        libcramjam::bzip2::compress(&mut Cursor::new(&input), &mut out, Some(6)).unwrap();
        times.push(t.elapsed());
        std::hint::black_box(out);
    }
    times.sort();
    let median = times[times.len() / 2];
    let throughput = input.len() as f64 / 1024.0 / 1024.0 / median.as_secs_f64();
    eprintln!("  median: {:?}  ({:.1} MB/s)", median, throughput);

    eprintln!("\n--- C bzip2 compress L6 ---");
    let mut times = Vec::new();
    for _ in 0..5 {
        let t = Instant::now();
        let mut enc = bzip2::read::BzEncoder::new(&input[..], bzip2::Compression::new(6));
        let mut out = Vec::new();
        enc.read_to_end(&mut out).unwrap();
        times.push(t.elapsed());
        std::hint::black_box(out);
    }
    times.sort();
    let median = times[times.len() / 2];
    let throughput = input.len() as f64 / 1024.0 / 1024.0 / median.as_secs_f64();
    eprintln!("  median: {:?}  ({:.1} MB/s)", median, throughput);
}
