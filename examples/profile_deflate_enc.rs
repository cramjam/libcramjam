//! Deflate encoder harness: ours vs flate2 (miniz_oxide) interleaved per
//! iteration, on the bench subset or FILE, at LEVELS (default 1,6,9).
//! Prints bytes and time ratios per level.
use std::io::{Cursor, Read, Write};
use std::time::{Duration, Instant};

fn load(name: &str) -> Vec<u8> {
    let bz2 = std::fs::read(format!("benches/data/{name}.bz2")).expect("read corpus");
    let mut out = Vec::new();
    bzip2::read::BzDecoder::new(&bz2[..]).read_to_end(&mut out).unwrap();
    out
}

const SUBSET: &[&str] = &[
    "Mark.Twain-Tom.Sawyer.txt", "html", "paper-100k.pdf", "fireworks.jpeg", "asyoulik.txt",
    "alice29.txt", "kppkn.gtb", "html_x_4", "lcet10.txt", "plrabn12.txt", "reymont",
];

fn ours(d: &[u8], level: u32) -> Vec<u8> {
    let mut o = Vec::with_capacity(d.len());
    libcramjam::deflate::compress(&mut Cursor::new(d), &mut o, Some(level)).unwrap();
    o
}
fn theirs(d: &[u8], level: u32) -> Vec<u8> {
    let mut e = flate2::write::DeflateEncoder::new(Vec::new(), flate2::Compression::new(level));
    e.write_all(d).unwrap();
    e.finish().unwrap()
}

fn main() {
    let files: Vec<(String, Vec<u8>)> = match std::env::var("FILE") {
        Ok(f) => vec![(f.clone(), load(&f))],
        Err(_) => SUBSET.iter().map(|f| (f.to_string(), load(f))).collect(),
    };
    let levels: Vec<u32> = std::env::var("LEVELS")
        .map(|s| s.split(',').map(|x| x.parse().unwrap()).collect())
        .unwrap_or(vec![1, 6, 9]);
    let iters: usize = std::env::var("ITERS").ok().and_then(|s| s.parse().ok()).unwrap_or(5);
    let total: usize = files.iter().map(|(_, d)| d.len()).sum();
    for &level in &levels {
        let (mut ob, mut tb) = (0usize, 0usize);
        let (mut ot, mut tt) = (Vec::new(), Vec::new());
        for _ in 0..iters {
            let mut o = Duration::ZERO;
            let mut t = Duration::ZERO;
            ob = 0;
            tb = 0;
            for (_, d) in &files {
                let t0 = Instant::now();
                let a = ours(d, level);
                o += t0.elapsed();
                let t0 = Instant::now();
                let b = theirs(d, level);
                t += t0.elapsed();
                ob += a.len();
                tb += b.len();
            }
            ot.push(o);
            tt.push(t);
        }
        ot.sort();
        tt.sort();
        let (o, t) = (ot[iters / 2], tt[iters / 2]);
        println!(
            "L{level}: bytes ours {ob} ({:.2}%) vs miniz {tb} ({:.2}%) = {:+.2}pp | time ours {:.1}ms vs miniz {:.1}ms = {:.2}x",
            ob as f64 * 100.0 / total as f64,
            tb as f64 * 100.0 / total as f64,
            (ob as f64 - tb as f64) * 100.0 / total as f64,
            o.as_secs_f64() * 1e3,
            t.as_secs_f64() * 1e3,
            o.as_secs_f64() / t.as_secs_f64()
        );
    }
}
