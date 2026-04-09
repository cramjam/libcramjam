//! Corpus-driven cross-impl tests for our pure-Rust xz/lzma vs the C `xz2` crate.
//!
//! See `corpus_zstd.rs` for the test layout — same shape, different codec.

use std::io::{Cursor, Read, Write};

#[path = "../benches/common.rs"]
mod common;

const LEVELS: &[u32] = &[1, 6, 9];
const TOL_PP: f64 = 10.0;

fn ours_compress(data: &[u8], level: u32) -> Vec<u8> {
    let mut out = Vec::with_capacity(data.len());
    libcramjam::xz::compress(
        &mut Cursor::new(data),
        &mut out,
        Some(level),
        None::<libcramjam::xz::Format>,
        None::<libcramjam::xz::Check>,
        None::<libcramjam::xz::Filters>,
        None::<libcramjam::xz::LzmaOptions>,
    )
    .unwrap();
    out
}

fn ours_decompress(data: &[u8]) -> Vec<u8> {
    let mut out = Vec::with_capacity(data.len() * 2);
    libcramjam::xz::decompress(&mut Cursor::new(data), &mut out).unwrap();
    out
}

fn c_compress(data: &[u8], level: u32) -> Vec<u8> {
    let mut enc = xz2::write::XzEncoder::new(Vec::new(), level);
    enc.write_all(data).unwrap();
    enc.finish().unwrap()
}

fn c_decompress(data: &[u8]) -> Vec<u8> {
    let mut dec = xz2::read::XzDecoder::new(data);
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

/// Full-corpus run.  Slow (~5 min release build, all 24 files × 3 levels)
/// so it's `#[ignore]`d by default — invoke explicitly with
/// `cargo test --release --test corpus_xz -- --ignored`.
#[test]
#[ignore = "full corpus is slow; run with --ignored"]
fn corpus_full() {
    for (name, data) in common::load_all() {
        check_corpus(name, data);
    }
}
