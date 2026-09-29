//! Fast-parser check: our block output vs `LZ4_compress_default` (byte
//! equality), plus frame sizes and interleaved timing vs the C frame encoder
//! at LEVEL (default 1) for the bench subset (or FILE).
#[path = "../benches/common.rs"]
#[allow(dead_code)]
mod common;
use std::io::{Cursor, Write};
use std::time::{Duration, Instant};

fn median(v: &mut Vec<Duration>) -> Duration { v.sort(); v[v.len() / 2] }

fn main() {
    let level: u32 = std::env::var("LEVEL").ok().and_then(|s| s.parse().ok()).unwrap_or(1);
    let iters: usize = std::env::var("ITERS").ok().and_then(|s| s.parse().ok()).unwrap_or(5);
    let files: Vec<(&str, &[u8])> = match std::env::var("FILE") {
        Ok(f) => vec![(Box::leak(f.clone().into_boxed_str()), common::load(Box::leak(f.into_boxed_str())))],
        Err(_) => common::load_bench_subset(),
    };
    let (mut to, mut tc, mut bo, mut bc) = (0f64, 0f64, 0usize, 0usize);
    let mut all_same = true;
    for (name, data) in files {
        // Block API byte comparison at two sizes: < 64 KiB (u16 table) and the full file (u32 table).
        let mut results = Vec::new();
        for blk in [&data[..data.len().min(60_000)], &data[..data.len().min(4 << 20)]] {
            let mut ours = Vec::new();
            libcramjam::lz4_impl_block_fast(blk, &mut ours);
            let theirs = lz4::block::compress(blk, None, false).unwrap();
            let first = ours.iter().zip(theirs.iter()).position(|(a, b)| a != b);
            let same = ours == theirs && ours.len() == theirs.len();
            all_same &= same;
            results.push(format!("{} ({} vs {}, first diff {:?})", if same { "IDENTICAL" } else { "DIFFERS" }, ours.len(), theirs.len(), first));
        }
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
        eprintln!("{name:>12} L{level}: block<64K {}  block {}  frame {} vs {} bytes  time {:.2?} vs {:.2?} = {:.2}x",
            results[0], results[1], fo, fc, o, c, o.as_secs_f64() / c.as_secs_f64());
    }
    eprintln!("TOTAL L{level}: blocks {}  frame bytes {bo} vs {bc} ({:+.3}%)  time {:.2}x",
        if all_same { "ALL IDENTICAL" } else { "SOME DIFFER" }, (bo as f64 - bc as f64) / bc as f64 * 100.0, to / tc);
}
