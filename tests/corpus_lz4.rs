//! Corpus-driven cross-impl tests for our pure-Rust lz4 vs the C `lz4` crate.
//!
//! For every file in the shared benchmark corpus, verifies:
//!   1. Our compressed output is within `TOL_PP` percentage points of C lz4's
//!      output ratio.
//!   2. C-compressed → ours-decompressed is byte-identical to the input.
//!   3. Ours-compressed → C-decompressed is byte-identical to the input.
//!
//! `libcramjam::lz4::compress` doesn't take a level argument (it uses the
//! frame format's default), so we don't iterate over levels here.

use std::io::{Cursor, Read, Write};

#[path = "../benches/common.rs"]
mod common;

const TOL_PP: f64 = 10.0;

fn ours_compress(data: &[u8]) -> Vec<u8> {
    let mut out = Vec::with_capacity(data.len());
    libcramjam::lz4::compress(&mut Cursor::new(data), &mut out, None).unwrap();
    out
}

fn ours_decompress(data: &[u8]) -> Vec<u8> {
    let mut out = Vec::with_capacity(data.len() * 2);
    libcramjam::lz4::decompress(&mut Cursor::new(data), &mut out).unwrap();
    out
}

fn c_compress(data: &[u8]) -> Vec<u8> {
    // Level 1 = fast mode in the lz4 frame format.  Our pure-Rust encoder is
    // fast-mode only (no HC), so comparing against HC (`.level(4+)`) would be
    // an unfair test of an algorithm we don't implement.
    let mut enc = lz4::EncoderBuilder::new()
        .level(1)
        .auto_flush(true)
        .build(Vec::new())
        .unwrap();
    enc.write_all(data).unwrap();
    let (out, r) = enc.finish();
    r.unwrap();
    out
}

fn c_decompress(data: &[u8]) -> Vec<u8> {
    let mut dec = lz4::Decoder::new(data).unwrap();
    let mut out = Vec::new();
    dec.read_to_end(&mut out).unwrap();
    out
}

fn check_corpus(name: &str, data: &[u8]) {
    let ours = ours_compress(data);
    let theirs = c_compress(data);

    let delta_pp =
        (ours.len() as f64 - theirs.len() as f64).abs() / data.len().max(1) as f64 * 100.0;
    assert!(
        delta_pp <= TOL_PP,
        "ratio gap too large: corpus={name} ours={} theirs={} delta={:.2}pp tol={:.1}pp",
        ours.len(),
        theirs.len(),
        delta_pp,
        TOL_PP,
    );

    // theirs -> ours
    let round1 = ours_decompress(&theirs);
    assert_eq!(round1.len(), data.len(), "c→ours length mismatch: corpus={name}");
    assert!(round1 == data, "c→ours bytes mismatch: corpus={name}");

    // ours -> theirs
    let round2 = c_decompress(&ours);
    assert_eq!(round2.len(), data.len(), "ours→c length mismatch: corpus={name}");
    assert!(round2 == data, "ours→c bytes mismatch: corpus={name}");
}

#[test]
fn corpus_subset() {
    for (name, data) in common::load_bench_subset() {
        check_corpus(name, data);
    }
}

#[test]
#[ignore = "full corpus is slow; run with --ignored"]
fn corpus_full() {
    for (name, data) in common::load_all() {
        check_corpus(name, data);
    }
}
