//! Manual time-instrumented profile of zstd compress and decompress.
use std::io::{Cursor, Read};
use std::time::Instant;

fn main() {
    let path = std::path::Path::new("benches/data/dickens.bz2");
    let bz2 = std::fs::read(path).expect("read corpus");
    let mut dec = bzip2::read::BzDecoder::new(&bz2[..]);
    let mut input = Vec::new();
    dec.read_to_end(&mut input).unwrap();
    eprintln!("Loaded {} bytes from dickens", input.len());

    eprintln!("\n--- ours zstd compress L3 ---");
    let mut times = Vec::new();
    for _ in 0..10 {
        let mut out = Vec::with_capacity(input.len());
        let t = Instant::now();
        libcramjam::zstd::compress(
            &mut Cursor::new(&input), &mut out, Some(3), Some(input.len()),
        ).unwrap();
        times.push(t.elapsed());
        std::hint::black_box(out);
    }
    times.sort();
    let median = times[times.len() / 2];
    let throughput = input.len() as f64 / 1024.0 / 1024.0 / median.as_secs_f64();
    eprintln!("  median: {:?}  ({:.0} MB/s)", median, throughput);

    eprintln!("\n--- C zstd compress L3 ---");
    let mut times = Vec::new();
    for _ in 0..10 {
        let t = Instant::now();
        let mut enc = zstd::stream::read::Encoder::new(&input[..], 3).unwrap();
        let mut out = Vec::new();
        enc.read_to_end(&mut out).unwrap();
        times.push(t.elapsed());
        std::hint::black_box(out);
    }
    times.sort();
    let median = times[times.len() / 2];
    let throughput = input.len() as f64 / 1024.0 / 1024.0 / median.as_secs_f64();
    eprintln!("  median: {:?}  ({:.0} MB/s)", median, throughput);

    let mut compressed = Vec::with_capacity(input.len());
    libcramjam::zstd::compress(
        &mut Cursor::new(&input), &mut compressed, Some(3), Some(input.len()),
    ).unwrap();

    eprintln!("\n--- ours zstd decompress ---");
    let mut times = Vec::new();
    for _ in 0..30 {
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

    eprintln!("\n--- C zstd decompress ---");
    let mut times = Vec::new();
    for _ in 0..30 {
        let t = Instant::now();
        let mut dec = zstd::stream::read::Decoder::new(&compressed[..]).unwrap();
        let mut out = Vec::new();
        dec.read_to_end(&mut out).unwrap();
        times.push(t.elapsed());
        std::hint::black_box(out);
    }
    times.sort();
    let median = times[times.len() / 2];
    let throughput = input.len() as f64 / 1024.0 / 1024.0 / median.as_secs_f64();
    eprintln!("  median: {:?}  ({:.0} MB/s)", median, throughput);
}
