//! xz compress harness: ours vs liblzma (xz2), interleaved so both see the
//! same machine load. Env: FILE (corpus name, default dickens), LEVELS
//! (comma list, default 1,6,9), ITERS (default 3).
use std::io::{Cursor, Read, Write};
use std::time::{Duration, Instant};

fn median(v: &mut Vec<Duration>) -> Duration {
    v.sort();
    v[v.len() / 2]
}

fn main() {
    let file = std::env::var("FILE").unwrap_or("dickens".into());
    let levels: Vec<u32> = std::env::var("LEVELS")
        .unwrap_or("1,6,9".into())
        .split(',')
        .map(|s| s.trim().parse().unwrap())
        .collect();
    let iters: usize = std::env::var("ITERS").ok().and_then(|s| s.parse().ok()).unwrap_or(3);
    let bz2 = std::fs::read(format!("benches/data/{file}.bz2")).expect("read corpus");
    let mut input = Vec::new();
    bzip2::read::BzDecoder::new(&bz2[..]).read_to_end(&mut input).unwrap();
    eprintln!("{file}: {} bytes", input.len());

    for &level in &levels {
        let mut ours = Vec::new();
        let mut theirs = Vec::new();
        let mut our_size = 0;
        let mut c_size = 0;
        for _ in 0..iters {
            let t = Instant::now();
            let mut out = Vec::new();
            libcramjam::xz::compress(
                &mut Cursor::new(&input),
                &mut out,
                Some(level),
                None::<libcramjam::xz::Format>,
                None::<libcramjam::xz::Check>,
                None::<libcramjam::xz::Filters>,
                None::<libcramjam::xz::LzmaOptions>,
            )
            .unwrap();
            ours.push(t.elapsed());
            our_size = out.len();
            // Round-trip through liblzma to prove validity.
            let mut back = Vec::new();
            xz2::read::XzDecoder::new(&out[..]).read_to_end(&mut back).unwrap();
            assert!(back == input, "liblzma decode mismatch at preset {level}");

            let t = Instant::now();
            let mut enc = xz2::write::XzEncoder::new(Vec::new(), level);
            enc.write_all(&input).unwrap();
            let c = enc.finish().unwrap();
            theirs.push(t.elapsed());
            c_size = c.len();
        }
        let m = median(&mut ours);
        let c = median(&mut theirs);
        eprintln!(
            "L{level}: ours {:>9.1?} {:>9} bytes ({:.2}%) | C {:>9.1?} {:>9} bytes ({:.2}%) | time {:.2}x  size {:+.2}pp",
            m,
            our_size,
            our_size as f64 * 100.0 / input.len() as f64,
            c,
            c_size,
            c_size as f64 * 100.0 / input.len() as f64,
            m.as_secs_f64() / c.as_secs_f64(),
            (our_size as f64 - c_size as f64) * 100.0 / input.len() as f64,
        );
    }
}
