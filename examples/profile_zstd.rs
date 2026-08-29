//! Compress profile harness for callgrind + A/B timing. Env: FILE (corpus
//! name, default dickens), LEVEL (default 3), ITERS (default 10), TIME_C=1
//! to time C zstd interleaved with ours (same-run ratios are the only
//! trustworthy numbers on a loaded machine).
use std::io::{Cursor, Read};
use std::time::{Duration, Instant};

fn median(v: &mut Vec<Duration>) -> Duration {
    v.sort();
    v[v.len() / 2]
}

fn main() {
    let file = std::env::var("FILE").unwrap_or("dickens".into());
    let level: i32 = std::env::var("LEVEL").ok().and_then(|s| s.parse().ok()).unwrap_or(3);
    let iters: usize = std::env::var("ITERS").ok().and_then(|s| s.parse().ok()).unwrap_or(10);
    let time_c = std::env::var_os("TIME_C").is_some();
    let bz2 = std::fs::read(format!("benches/data/{file}.bz2")).expect("read corpus");
    let mut input = Vec::new();
    bzip2::read::BzDecoder::new(&bz2[..]).read_to_end(&mut input).unwrap();
    eprintln!("Loaded {} bytes from {file}, level {level}", input.len());

    let mut ours = Vec::new();
    let mut theirs = Vec::new();
    let mut ours_size = 0;
    let mut c_size = 0;
    for _ in 0..iters {
        let mut out = Vec::with_capacity(input.len());
        let t = Instant::now();
        libcramjam::zstd::compress(&mut Cursor::new(&input), &mut out, Some(level), Some(input.len())).unwrap();
        ours.push(t.elapsed());
        ours_size = out.len();
        std::hint::black_box(out);
        if time_c {
            let t = Instant::now();
            let mut enc = zstd::stream::read::Encoder::new(&input[..], level).unwrap();
            let mut out = Vec::new();
            enc.read_to_end(&mut out).unwrap();
            theirs.push(t.elapsed());
            c_size = out.len();
            std::hint::black_box(out);
        }
    }
    let mb = input.len() as f64 / 1048576.0;
    let m = median(&mut ours);
    eprintln!(
        "ours: median {:?} ({:.0} MB/s) size {} ({:.2}%)",
        m,
        mb / m.as_secs_f64(),
        ours_size,
        100.0 * ours_size as f64 / input.len() as f64
    );
    if time_c {
        let c = median(&mut theirs);
        eprintln!(
            "C:    median {:?} ({:.0} MB/s) size {} ({:.2}%)  ours/C = {:.2}x",
            c,
            mb / c.as_secs_f64(),
            c_size,
            100.0 * c_size as f64 / input.len() as f64,
            m.as_secs_f64() / c.as_secs_f64()
        );
    }
}
