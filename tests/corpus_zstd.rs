//! Corpus-driven cross-impl tests for our pure-Rust zstd vs the C `zstd` crate.
//!
//! For every file in the shared benchmark corpus and every configured level,
//! verifies:
//!   1. Our compressed output is within `TOL_PP` percentage points of C zstd's
//!      output ratio (ours/input ≈ theirs/input).
//!   2. C-compressed → ours-decompressed is byte-identical to the input.
//!   3. Ours-compressed → C-decompressed is byte-identical to the input.
//!
//! The default test runs on the BENCH_SUBSET (smaller files, fast feedback).
//! A `#[ignore]`d test exercises the full corpus on opt-in
//! (`cargo test --release --test corpus_zstd -- --ignored`).

use std::io::{Cursor, Read};

#[path = "../benches/common.rs"]
mod common;

const LEVELS: &[i32] = &[1, 3, 6, 9];

/// Allowed gap between our compressed-ratio and C zstd's, in percentage
/// points of the input size.  Worst case after custom-FSE + package-merge +
/// hash/chain tuning is kppkn at level 1 (~8.2 pp).  Keep 10 pp as a
/// regression net with headroom.
const TOL_PP: f64 = 10.0;

fn ours_compress(data: &[u8], level: i32) -> Vec<u8> {
    let mut out = Vec::with_capacity(data.len());
    libcramjam::zstd::compress(&mut Cursor::new(data), &mut out, Some(level), Some(data.len())).unwrap();
    out
}

fn ours_decompress(data: &[u8]) -> Vec<u8> {
    let mut out = Vec::with_capacity(data.len() * 2);
    libcramjam::zstd::decompress(&mut Cursor::new(data), &mut out).unwrap();
    out
}

fn c_compress(data: &[u8], level: i32) -> Vec<u8> {
    let mut enc = zstd::stream::read::Encoder::new(data, level).unwrap();
    let mut out = Vec::new();
    enc.read_to_end(&mut out).unwrap();
    out
}

fn c_decompress(data: &[u8]) -> Vec<u8> {
    let mut dec = zstd::stream::read::Decoder::new(data).unwrap();
    let mut out = Vec::new();
    dec.read_to_end(&mut out).unwrap();
    out
}

fn check_corpus(name: &str, data: &[u8], levels: &[i32]) {
    for &level in levels {
        let ours = ours_compress(data, level);
        let theirs = c_compress(data, level);

        // 1. Ratio is within tolerance.
        let delta_pp =
            (ours.len() as f64 - theirs.len() as f64).abs() / data.len() as f64 * 100.0;
        assert!(
            delta_pp <= TOL_PP,
            "ratio gap too large: corpus={name} level={level} ours={} theirs={} delta={:.2}pp tol={:.1}pp",
            ours.len(),
            theirs.len(),
            delta_pp,
            TOL_PP,
        );

        // 2. theirs -> ours decode.
        let round1 = ours_decompress(&theirs);
        assert_eq!(
            round1.len(),
            data.len(),
            "c→ours length mismatch: corpus={name} level={level}"
        );
        assert!(
            round1 == data,
            "c→ours bytes mismatch: corpus={name} level={level}"
        );

        // 3. ours -> theirs decode.
        let round2 = c_decompress(&ours);
        assert_eq!(
            round2.len(),
            data.len(),
            "ours→c length mismatch: corpus={name} level={level}"
        );
        assert!(
            round2 == data,
            "ours→c bytes mismatch: corpus={name} level={level}"
        );
    }
}

#[test]
fn corpus_subset() {
    for (name, data) in common::load_bench_subset() {
        check_corpus(name, data, LEVELS);
    }
}

#[test]
#[ignore = "full corpus is slow; run with --ignored"]
fn corpus_full() {
    for (name, data) in common::load_all() {
        check_corpus(name, data, LEVELS);
    }
}
