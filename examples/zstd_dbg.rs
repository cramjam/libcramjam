//! Debug repro: C-compress a corpus file, decode with ours, report the first mismatch.
use std::io::{Cursor, Read};
fn main() {
    let name = std::env::args().nth(1).unwrap_or("kppkn.gtb".into());
    let level: i32 = std::env::args().nth(2).and_then(|s| s.parse().ok()).unwrap_or(1);
    let bz2 = std::fs::read(format!("benches/data/{name}.bz2")).unwrap();
    let mut input = Vec::new();
    bzip2::read::BzDecoder::new(&bz2[..]).read_to_end(&mut input).unwrap();
    let mut comp = Vec::new();
    zstd::stream::read::Encoder::new(&input[..], level).unwrap().read_to_end(&mut comp).unwrap();
    let mut out = Vec::new();
    libcramjam::zstd::decompress(&mut Cursor::new(&comp), &mut out).unwrap();
    let first = out.iter().zip(input.iter()).position(|(a, b)| a != b);
    eprintln!("len ours={} exp={} first_mismatch={:?}", out.len(), input.len(), first);
}
