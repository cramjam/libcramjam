//! Prints a checksum + length of our bzip2 output for a few corpus files and
//! levels — used to verify encoder changes are byte-identical.
//! `CORPUS` env var overrides the corpus directory (default `benches/data`).
use std::io::{Cursor, Read};

fn main() {
    let dir = std::env::var("CORPUS").unwrap_or("benches/data".into());
    for file in ["dickens", "kppkn.gtb", "fireworks.jpeg", "xml", "alice29.txt"] {
        let bz2 = std::fs::read(format!("{dir}/{file}.bz2")).unwrap();
        let mut input = Vec::new();
        bzip2::read::BzDecoder::new(&bz2[..]).read_to_end(&mut input).unwrap();
        for level in [1u32, 6, 9] {
            let mut out = Vec::new();
            libcramjam::bzip2::compress(&mut Cursor::new(&input), &mut out, Some(level)).unwrap();
            let h = out.iter().fold(0xcbf29ce484222325u64, |h, &b| (h ^ b as u64).wrapping_mul(0x100000001b3));
            println!("{file} L{level} len={} fnv={h:016x}", out.len());
        }
    }
}
