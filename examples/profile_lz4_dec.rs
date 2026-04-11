//! Decode-only lz4 profile harness (for callgrind).
use std::io::{Cursor, Read};
use std::time::Instant;

fn load_bz2(path: &str) -> Vec<u8> {
    let bz2 = std::fs::read(path).expect("read corpus");
    let mut dec = bzip2::read::BzDecoder::new(&bz2[..]);
    let mut out = Vec::new();
    dec.read_to_end(&mut out).unwrap();
    out
}

fn main() {
    let input = load_bz2("benches/data/dickens.bz2");
    eprintln!("Loaded {} bytes from dickens", input.len());

    // Produce a C-lz4-compressed payload once so we only profile decode.
    let compressed = {
        let mut buf = Vec::new();
        let mut enc = lz4::EncoderBuilder::new().level(4).build(&mut buf).unwrap();
        std::io::copy(&mut &input[..], &mut enc).unwrap();
        let (_, r) = enc.finish();
        r.unwrap();
        buf
    };
    eprintln!("Compressed: {} bytes", compressed.len());

    eprintln!("\n--- ours lz4 decompress (100 iters) ---");
    let mut times = Vec::new();
    for _ in 0..100 {
        let mut out = Vec::with_capacity(input.len());
        let t = Instant::now();
        libcramjam::lz4::decompress(&mut Cursor::new(&compressed), &mut out).unwrap();
        times.push(t.elapsed());
        std::hint::black_box(out);
    }
    times.sort();
    let median = times[times.len() / 2];
    let throughput = input.len() as f64 / 1024.0 / 1024.0 / median.as_secs_f64();
    eprintln!("  median: {:?}  ({:.0} MB/s)", median, throughput);
}
