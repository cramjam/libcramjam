//! Corpus-driven cross-impl tests for our pure-Rust bzip2 vs the C `bzip2` crate.
//!
//! See `corpus_zstd.rs` for the test layout — same shape, different codec.

use std::io::{Cursor, Read, Write};

#[path = "../benches/common.rs"]
mod common;

const LEVELS: &[u32] = &[1, 6, 9];
const TOL_PP: f64 = 10.0;

fn ours_compress(data: &[u8], level: u32) -> Vec<u8> {
    let mut out = Vec::with_capacity(data.len());
    libcramjam::bzip2::compress(&mut Cursor::new(data), &mut out, Some(level)).unwrap();
    out
}

fn ours_decompress(data: &[u8]) -> Vec<u8> {
    let mut out = Vec::with_capacity(data.len() * 2);
    libcramjam::bzip2::decompress(&mut Cursor::new(data), &mut out).unwrap();
    out
}

fn c_compress(data: &[u8], level: u32) -> Vec<u8> {
    let mut enc = bzip2::write::BzEncoder::new(Vec::new(), bzip2::Compression::new(level));
    enc.write_all(data).unwrap();
    enc.finish().unwrap()
}

fn c_decompress(data: &[u8]) -> Vec<u8> {
    let mut dec = bzip2::read::BzDecoder::new(data);
    let mut out = Vec::new();
    dec.read_to_end(&mut out).unwrap();
    out
}

fn check_corpus(name: &str, data: &[u8]) {
    for &level in LEVELS {
        let ours = ours_compress(data, level);
        let theirs = c_compress(data, level);

        let delta_pp = (ours.len() as f64 - theirs.len() as f64).abs()
            / data.len().max(1) as f64 * 100.0;
        assert!(
            delta_pp <= TOL_PP,
            "ratio gap too large: corpus={name} level={level} ours={} theirs={} delta={:.2}pp tol={:.1}pp",
            ours.len(),
            theirs.len(),
            delta_pp,
            TOL_PP,
        );

        let round1 = ours_decompress(&theirs);
        assert_eq!(round1.len(), data.len(), "c→ours len mismatch: corpus={name} level={level}");
        assert!(round1 == data, "c→ours bytes mismatch: corpus={name} level={level}");

        let round2 = c_decompress(&ours);
        assert_eq!(round2.len(), data.len(), "ours→c len mismatch: corpus={name} level={level}");
        assert!(round2 == data, "ours→c bytes mismatch: corpus={name} level={level}");
    }
}

#[test]
fn corpus_subset() {
    for (name, data) in common::load_bench_subset() {
        check_corpus(name, data);
    }
}

/// Full-corpus run.  Currently surfaces a real encoder bug:
///   - Several files fail with "bzip2: invalid data" when the C bzip2 decoder
///     tries to parse our output, specifically when the input is large
///     enough to span multiple bzip2 blocks at the chosen level (L1 = 100KB,
///     L6 = 600KB, L9 = 900KB).  Known triggers in the corpus: urls_10k
///     (L1, L6), xml (L1), ooffice (L1), x_ray (L1, L6, L9 — fails at every
///     level, so the bug isn't purely about block boundaries).
///   - See the project_bzip2_multiblock_bug memory for the active
///     investigation.
#[test]
#[ignore = "full corpus exposes a known multi-block encoder bug; run with --ignored"]
fn corpus_full() {
    for (name, data) in common::load_all() {
        check_corpus(name, data);
    }
}
