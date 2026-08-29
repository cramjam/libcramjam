//! Decode-only profile harness for callgrind + A/B timing. Pre-computes a
//! C-zstd-compressed frame for dickens (or FILE env var), then measures
//! ours and, with TIME_C=1, the C decoder interleaved so both see the same
//! machine load. ITERS overrides the iteration count (default 50).
use std::io::{Cursor, Read};
use std::time::{Duration, Instant};

fn load_bz2(path: &str) -> Vec<u8> {
    let bz2 = std::fs::read(path).expect("read corpus");
    let mut dec = bzip2::read::BzDecoder::new(&bz2[..]);
    let mut out = Vec::new();
    dec.read_to_end(&mut out).unwrap();
    out
}

fn median(v: &mut Vec<Duration>) -> Duration {
    v.sort();
    v[v.len() / 2]
}

fn main() {
    let file = std::env::var("FILE").unwrap_or("dickens".into());
    let level: i32 = std::env::var("LEVEL").ok().and_then(|s| s.parse().ok()).unwrap_or(3);
    let iters: usize = std::env::var("ITERS").ok().and_then(|s| s.parse().ok()).unwrap_or(50);
    let time_c = std::env::var_os("TIME_C").is_some();
    let input = load_bz2(&format!("benches/data/{file}.bz2"));
    eprintln!("Loaded {} bytes from {file}", input.len());

    let compressed = {
        let mut enc = zstd::stream::read::Encoder::new(&input[..], level).unwrap();
        let mut out = Vec::new();
        enc.read_to_end(&mut out).unwrap();
        out
    };
    eprintln!("Compressed: {} bytes (C zstd L{level})", compressed.len());

    let mut ours = Vec::new();
    let mut theirs = Vec::new();
    for _ in 0..iters {
        let mut out = Vec::with_capacity(input.len());
        let t = Instant::now();
        libcramjam::zstd::decompress(&mut Cursor::new(&compressed), &mut out).unwrap();
        ours.push(t.elapsed());
        assert_eq!(out.len(), input.len());
        std::hint::black_box(out);
        if time_c {
            let mut out = Vec::with_capacity(input.len());
            let t = Instant::now();
            zstd::stream::read::Decoder::new(&compressed[..]).unwrap().read_to_end(&mut out).unwrap();
            theirs.push(t.elapsed());
            std::hint::black_box(out);
        }
    }
    let m = median(&mut ours);
    let mb = input.len() as f64 / 1048576.0;
    eprintln!("ours: median {:?} ({:.0} MB/s)", m, mb / m.as_secs_f64());
    if time_c {
        let c = median(&mut theirs);
        eprintln!("C:    median {:?} ({:.0} MB/s)  ours/C = {:.2}x", c, mb / c.as_secs_f64(), m.as_secs_f64() / c.as_secs_f64());
    }
    eprintln!("  median: {:?}", m);
}
