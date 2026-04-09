//! Corpus-driven cross-impl tests for deflate / gzip / zlib vs `flate2`.
//!
//! For every file in the shared benchmark corpus and every level in
//! `LEVELS`, verifies for each of the three deflate-family wrappers:
//!   1. Our compressed output is within `TOL_PP` percentage points of
//!      flate2's output ratio.
//!   2. flate2-compressed → ours-decompressed roundtrips.
//!   3. ours-compressed → flate2-decompressed roundtrips.

use std::io::{Cursor, Read, Write};

#[path = "../benches/common.rs"]
mod common;

const LEVELS: &[u32] = &[1, 6, 9];
/// Allowed ratio gap vs flate2 in percentage points of input size.
/// Matches the user's "≤10%" target.  L6/L9 are typically within 1-2pp;
/// L1 is within ~9pp on the worst (tiny text) corpus file.
const TOL_PP: f64 = 10.0;

// ---------- ours ----------
fn ours_def_c(d: &[u8], l: u32) -> Vec<u8> {
    let mut o = Vec::with_capacity(d.len());
    libcramjam::deflate::compress(&mut Cursor::new(d), &mut o, Some(l)).unwrap();
    o
}
fn ours_def_d(d: &[u8]) -> Vec<u8> {
    let mut o = Vec::with_capacity(d.len() * 2);
    libcramjam::deflate::decompress(&mut Cursor::new(d), &mut o).unwrap();
    o
}
fn ours_gz_c(d: &[u8], l: u32) -> Vec<u8> {
    let mut o = Vec::with_capacity(d.len());
    libcramjam::gzip::compress(&mut Cursor::new(d), &mut o, Some(l)).unwrap();
    o
}
fn ours_gz_d(d: &[u8]) -> Vec<u8> {
    let mut o = Vec::with_capacity(d.len() * 2);
    libcramjam::gzip::decompress(&mut Cursor::new(d), &mut o).unwrap();
    o
}
fn ours_zl_c(d: &[u8], l: u32) -> Vec<u8> {
    let mut o = Vec::with_capacity(d.len());
    libcramjam::zlib::compress(&mut Cursor::new(d), &mut o, Some(l)).unwrap();
    o
}
fn ours_zl_d(d: &[u8]) -> Vec<u8> {
    let mut o = Vec::with_capacity(d.len() * 2);
    libcramjam::zlib::decompress(&mut Cursor::new(d), &mut o).unwrap();
    o
}

// ---------- flate2 ----------
fn f2_def_c(d: &[u8], l: u32) -> Vec<u8> {
    let mut e = flate2::write::DeflateEncoder::new(Vec::new(), flate2::Compression::new(l));
    e.write_all(d).unwrap();
    e.finish().unwrap()
}
fn f2_def_d(d: &[u8]) -> Vec<u8> {
    let mut dec = flate2::read::DeflateDecoder::new(d);
    let mut o = Vec::new();
    dec.read_to_end(&mut o).unwrap();
    o
}
fn f2_gz_c(d: &[u8], l: u32) -> Vec<u8> {
    let mut e = flate2::write::GzEncoder::new(Vec::new(), flate2::Compression::new(l));
    e.write_all(d).unwrap();
    e.finish().unwrap()
}
fn f2_gz_d(d: &[u8]) -> Vec<u8> {
    let mut dec = flate2::read::MultiGzDecoder::new(d);
    let mut o = Vec::new();
    dec.read_to_end(&mut o).unwrap();
    o
}
fn f2_zl_c(d: &[u8], l: u32) -> Vec<u8> {
    let mut e = flate2::write::ZlibEncoder::new(Vec::new(), flate2::Compression::new(l));
    e.write_all(d).unwrap();
    e.finish().unwrap()
}
fn f2_zl_d(d: &[u8]) -> Vec<u8> {
    let mut dec = flate2::read::ZlibDecoder::new(d);
    let mut o = Vec::new();
    dec.read_to_end(&mut o).unwrap();
    o
}

/// Generic per-codec roundtrip + ratio check.  Each function pointer pair
/// must be {ours, flate2}'s {compress(level), decompress} for the same
/// wrapper type — passing them in keeps the call sites uniform without
/// reaching for trait gymnastics.
fn check_codec(
    codec: &str,
    name: &str,
    data: &[u8],
    ours_c: fn(&[u8], u32) -> Vec<u8>,
    ours_d: fn(&[u8]) -> Vec<u8>,
    them_c: fn(&[u8], u32) -> Vec<u8>,
    them_d: fn(&[u8]) -> Vec<u8>,
) {
    for &level in LEVELS {
        let ours = ours_c(data, level);
        let theirs = them_c(data, level);

        let delta_pp = (ours.len() as f64 - theirs.len() as f64).abs()
            / data.len().max(1) as f64 * 100.0;
        assert!(
            delta_pp <= TOL_PP,
            "ratio gap too large: codec={codec} corpus={name} level={level} ours={} theirs={} delta={:.2}pp tol={:.1}pp",
            ours.len(),
            theirs.len(),
            delta_pp,
            TOL_PP,
        );

        let round1 = ours_d(&theirs);
        assert_eq!(round1.len(), data.len(),
            "flate2→ours len mismatch: codec={codec} corpus={name} level={level}");
        assert!(round1 == data,
            "flate2→ours bytes mismatch: codec={codec} corpus={name} level={level}");

        let round2 = them_d(&ours);
        assert_eq!(round2.len(), data.len(),
            "ours→flate2 len mismatch: codec={codec} corpus={name} level={level}");
        assert!(round2 == data,
            "ours→flate2 bytes mismatch: codec={codec} corpus={name} level={level}");
    }
}

fn check_all_codecs(name: &str, data: &[u8]) {
    check_codec("deflate", name, data, ours_def_c, ours_def_d, f2_def_c, f2_def_d);
    check_codec("gzip",    name, data, ours_gz_c,  ours_gz_d,  f2_gz_c,  f2_gz_d);
    check_codec("zlib",    name, data, ours_zl_c,  ours_zl_d,  f2_zl_c,  f2_zl_d);
}

#[test]
fn corpus_subset() {
    for (name, data) in common::load_bench_subset() {
        check_all_codecs(name, data);
    }
}

#[test]
#[ignore = "full corpus is slow; run with --ignored"]
fn corpus_full() {
    for (name, data) in common::load_all() {
        check_all_codecs(name, data);
    }
}
