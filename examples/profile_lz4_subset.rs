//! Mimics the summary bench's lz4 decode measurement (BENCH_SUBSET, C lz4 L4
//! frames, `Vec::new()` outputs) but interleaves ours/C per iteration so
//! the ratio survives a loaded machine. Also times the raw block decoder
//! into a pre-sized Vec to separate wrapper overhead from the hot loop.
#[path = "../benches/common.rs"]
#[allow(dead_code)]
mod common;

use std::io::{Cursor, Read, Write};
use std::time::{Duration, Instant};

fn median(v: &mut Vec<Duration>) -> Duration {
    v.sort();
    v[v.len() / 2]
}

fn main() {
    let files = common::load_bench_subset();
    let iters: usize = std::env::var("ITERS").ok().and_then(|s| s.parse().ok()).unwrap_or(7);
    let (mut tot_ours, mut tot_c, mut tot_raw, mut tot_rawc, mut tot_bytes) = (0f64, 0f64, 0f64, 0f64, 0usize);
    for (name, data) in &files {
        let compressed = {
            let mut enc = lz4::EncoderBuilder::new().level(4).build(Vec::new()).unwrap();
            enc.write_all(data).unwrap();
            let (out, r) = enc.finish();
            r.unwrap();
            out
        };
        let block = lz4::block::compress(data, Some(lz4::block::CompressionMode::HIGHCOMPRESSION(4)), true).unwrap();
        let (mut ours, mut theirs, mut raw, mut rawc) = (Vec::new(), Vec::new(), Vec::new(), Vec::new());
        for _ in 0..iters {
            let t = Instant::now();
            let mut out = Vec::new();
            libcramjam::lz4::decompress(&mut Cursor::new(&compressed), &mut out).unwrap();
            ours.push(t.elapsed());
            assert_eq!(out.len(), data.len());

            let t = Instant::now();
            let mut out = Vec::new();
            lz4::Decoder::new(&compressed[..]).unwrap().read_to_end(&mut out).unwrap();
            theirs.push(t.elapsed());
            assert_eq!(out.len(), data.len());

            // Block API: pure hot-loop comparison (what cramjam's
            // decompress_block uses).
            let t = Instant::now();
            let out = libcramjam::lz4::block::decompress_vec(&block).unwrap();
            raw.push(t.elapsed());
            assert_eq!(out.len(), data.len());
            let t = Instant::now();
            let out = lz4::block::decompress(&block, None).unwrap();
            rawc.push(t.elapsed());
            assert_eq!(out.len(), data.len());
        }
        let (o, c, r, rc) = (median(&mut ours), median(&mut theirs), median(&mut raw), median(&mut rawc));
        eprintln!("{name:>12}: frame ours {:>8.2?}  C {:>8.2?}  ratio {:.2}x   block ours {:>8.2?}  C {:>8.2?}  ratio {:.2}x",
            o, c, o.as_secs_f64() / c.as_secs_f64(), r, rc, r.as_secs_f64() / rc.as_secs_f64());
        tot_ours += o.as_secs_f64();
        tot_c += c.as_secs_f64();
        tot_raw += r.as_secs_f64();
        tot_rawc += rc.as_secs_f64();
        tot_bytes += data.len();
    }
    eprintln!("TOTAL {:.1} MB: frame ours {:.1}ms  C {:.1}ms  ratio {:.2}x   block ours {:.1}ms  C {:.1}ms  ratio {:.2}x",
        tot_bytes as f64 / 1048576.0, tot_ours * 1e3, tot_c * 1e3, tot_ours / tot_c, tot_raw * 1e3, tot_rawc * 1e3, tot_raw / tot_rawc);
}
