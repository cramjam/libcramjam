//! HC encoder check: our block/frame output vs the C reference at LEVEL for
//! the bench subset (or FILE): byte equality of the single-block output
//! (`lz4::block::compress` HC), frame sizes, and interleaved timing.
#[path = "../benches/common.rs"]
#[allow(dead_code)]
mod common;
use std::io::{Cursor, Write};
use std::time::{Duration, Instant};

fn median(v: &mut Vec<Duration>) -> Duration { v.sort(); v[v.len() / 2] }

fn main() {
    let level: u32 = std::env::var("LEVEL").ok().and_then(|s| s.parse().ok()).unwrap_or(9);
    let iters: usize = std::env::var("ITERS").ok().and_then(|s| s.parse().ok()).unwrap_or(3);
    let files: Vec<(&str, &[u8])> = match std::env::var("FILE") {
        Ok(f) => vec![(Box::leak(f.clone().into_boxed_str()), common::load(Box::leak(f.into_boxed_str())))],
        Err(_) => common::load_bench_subset(),
    };
    let (mut to, mut tc, mut bo, mut bc) = (0f64, 0f64, 0usize, 0usize);
    for (name, data) in files {
        // Single-block byte comparison (input capped to 4 MiB blocks by the C API? use up to 1 MiB).
        let blk = &data[..data.len().min(1 << 20)];
        let mut ours = Vec::new();
        libcramjam::lz4_impl_block_hc(blk, &mut ours, level);
        let theirs = lz4::block::compress(blk, Some(lz4::block::CompressionMode::HIGHCOMPRESSION(level as i32)), false).unwrap();
        let first = ours.iter().zip(theirs.iter()).position(|(a, b)| a != b);
        let same = ours == theirs;
        // Frame timing + sizes.
        let (mut t_o, mut t_c) = (Vec::new(), Vec::new());
        let (mut fo, mut fc) = (0, 0);
        for _ in 0..iters {
            let t = Instant::now();
            let mut out = Vec::new();
            libcramjam::lz4::compress(&mut Cursor::new(data), &mut out, Some(level)).unwrap();
            t_o.push(t.elapsed());
            fo = out.len();
            let t = Instant::now();
            let mut enc = lz4::EncoderBuilder::new().level(level).auto_flush(true).build(Vec::new()).unwrap();
            enc.write_all(data).unwrap();
            let (out, r) = enc.finish();
            r.unwrap();
            t_c.push(t.elapsed());
            fc = out.len();
        }
        let (o, c) = (median(&mut t_o), median(&mut t_c));
        to += o.as_secs_f64(); tc += c.as_secs_f64(); bo += fo; bc += fc;
        eprintln!("{name:>12} L{level}: block {} ({} vs {} bytes, first diff {:?})  frame {} vs {} bytes  time {:.2?} vs {:.2?} = {:.2}x",
            if same { "IDENTICAL" } else { "DIFFERS" }, ours.len(), theirs.len(), first, fo, fc, o, c, o.as_secs_f64() / c.as_secs_f64());
    }
    eprintln!("TOTAL L{level}: frame bytes {bo} vs {bc} ({:+.2}pp)  time {:.2}x", (bo as f64 - bc as f64) / 1.0 / (bo as f64 / 100.0).max(1.0), to / tc);
}
