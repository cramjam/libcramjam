//! Per-call overhead on small inputs: our cramjam entry point vs the C APIs.
#[path = "../benches/common.rs"]
#[allow(dead_code)]
mod common;
use std::io::Cursor;
use std::time::Instant;
fn best(mut f: impl FnMut(), n: usize) -> f64 {
    let mut b = f64::MAX;
    for _ in 0..n { let t = Instant::now(); f(); b = b.min(t.elapsed().as_secs_f64() * 1e6); }
    b
}
fn main() {
    let file = std::env::var("FILE").unwrap_or("Mark.Twain-Tom.Sawyer.txt".into());
    let data = common::load(Box::leak(file.into_boxed_str()));
    let n = 2000;
    if let Ok(mode) = std::env::var("MODE") {
        let iters: usize = std::env::var("ITERS").ok().and_then(|s| s.parse().ok()).unwrap_or(50);
        let mut out = Vec::new();
        for _ in 0..iters {
            out = Vec::new();
            if mode == "ours" {
                libcramjam::zstd::compress(&mut Cursor::new(data), &mut Cursor::new(&mut out), Some(3), Some(data.len())).unwrap();
            } else {
                zstd::stream::copy_encode(data, &mut Cursor::new(&mut out), 3).unwrap();
            }
        }
        println!("{} {}", mode, out.len());
        return;
    }
    let mut out = Vec::new();
    println!("{} bytes", data.len());
    println!("ours cramjam path (Read/Write):   {:7.1} us", best(|| { out.clear(); libcramjam::zstd::compress(&mut Cursor::new(data), &mut Cursor::new(&mut out), Some(3), Some(data.len())).unwrap(); }, n));
    println!("ours compress_bytes:              {:7.1} us", best(|| { out = libcramjam::zstd::compress_bytes(data, Some(3)); }, n));
    println!("C copy_encode (cramjam C path):   {:7.1} us", best(|| { out.clear(); zstd::stream::copy_encode(data, &mut Cursor::new(&mut out), 3).unwrap(); }, n));
    println!("C encode_all:                     {:7.1} us", best(|| { out = zstd::stream::encode_all(data, 3).unwrap(); }, n));
    println!("C bulk::compress:                 {:7.1} us", best(|| { out = zstd::bulk::compress(data, 3).unwrap(); }, n));
    let c = zstd::bulk::compress(data, 3).unwrap();
    let mut dec = Vec::new();
    println!("ours decompress (Read/Write):     {:7.1} us", best(|| { dec.clear(); libcramjam::zstd::decompress(&c[..], &mut Cursor::new(&mut dec)).unwrap(); }, n));
    println!("C copy_decode:                    {:7.1} us", best(|| { dec.clear(); zstd::stream::copy_decode(&c[..], &mut Cursor::new(&mut dec)).unwrap(); }, n));
}
