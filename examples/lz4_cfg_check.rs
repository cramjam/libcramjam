//! Bisect the C lz4 frame encoder settings released cramjam used.
#[path = "../benches/common.rs"]
#[allow(dead_code)]
mod common;
use std::io::{BufReader, Cursor, Write};
use std::time::Instant;

fn run(data: &[u8], chunk: usize, auto: bool, fav: bool, level: u32) -> (f64, usize) {
    let mut best = f64::MAX;
    let mut size = 0;
    for _ in 0..3 {
        let t = Instant::now();
        let mut enc = lz4::EncoderBuilder::new().level(level).auto_flush(auto).favor_dec_speed(fav).build(Vec::new()).unwrap();
        for c in data.chunks(chunk) { enc.write_all(c).unwrap(); }
        let (out, r) = enc.finish();
        r.unwrap();
        best = best.min(t.elapsed().as_secs_f64() * 1e3);
        size = out.len();
    }
    (best, size)
}

fn main() {
    let level: u32 = std::env::var("LEVEL").ok().and_then(|s| s.parse().ok()).unwrap_or(4);
    let file = std::env::var("FILE").unwrap_or("x_ray".into());
    let data = common::load(Box::leak(file.into_boxed_str()));
    for (chunk, auto, fav) in [(8192, true, true), (8192, true, false), (65536, true, true), (data.len(), true, true), (data.len(), false, true), (8192, false, true)] {
        let (ms, size) = run(data, chunk, auto, fav, level);
        println!("chunk {chunk:>9} auto_flush {auto:5} favor_dec {fav:5}: {ms:8.1} ms  {size} bytes");
    }
    // cramjam-main shape exactly: BufReader + io::copy
    let t = Instant::now();
    let mut enc = lz4::EncoderBuilder::new().level(level).auto_flush(true).favor_dec_speed(true).build(Vec::new()).unwrap();
    std::io::copy(&mut BufReader::new(Cursor::new(data)), &mut enc).unwrap();
    let (out, r) = enc.finish(); r.unwrap();
    println!("cramjam-main shape (BufReader+io::copy): {:.1} ms  {} bytes", t.elapsed().as_secs_f64()*1e3, out.len());
    let t = Instant::now();
    let mut ours = Vec::new();
    libcramjam::lz4::compress(&mut Cursor::new(data), &mut ours, Some(level)).unwrap();
    println!("ours: {:.1} ms  {} bytes", t.elapsed().as_secs_f64()*1e3, ours.len());
}
