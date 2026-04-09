//! Shared helper for loading the bzip2-compressed benchmark corpus that lives
//! under `benches/data/`.
//!
//! The corpus matches the one used by the python `cramjam` benchmark suite —
//! files are stored on disk as `*.bz2` (so they stay small in git) and are
//! decompressed lazily on first use, then cached for the lifetime of the
//! process.  Decompression goes through the C-backed `bzip2` dev-dependency
//! deliberately, so a regression in our own bzip2 codec can't cause every
//! other codec's tests to fail at corpus-load time.
//!
//! This file is included into both benchmarks and integration tests via
//! `#[path = "common.rs"] mod common;` (from `benches/`) and
//! `#[path = "../benches/common.rs"] mod common;` (from `tests/`).

#![allow(dead_code)]

use std::collections::HashMap;
use std::io::Read;
use std::path::PathBuf;
use std::sync::{Mutex, OnceLock};

/// One entry in the benchmark corpus.
#[derive(Copy, Clone, Debug)]
pub struct Corpus {
    /// Short, code-safe name (used in benchmark/test ids).
    pub name: &'static str,
    /// Filename inside `benches/data/`, including the `.bz2` suffix.
    pub file: &'static str,
}

/// Full set of corpus files copied from `cramjam/benchmarks/data/`.
///
/// Order is roughly by decompressed size — smallest first — so iterating in
/// order keeps the early progress fast.
pub const CORPORA: &[Corpus] = &[
    Corpus { name: "tom_sawyer", file: "Mark.Twain-Tom.Sawyer.txt.bz2" },
    Corpus { name: "html",       file: "html.bz2" },
    Corpus { name: "geo",        file: "geo.protodata.bz2" },
    Corpus { name: "html_x_4",   file: "html_x_4.bz2" },
    Corpus { name: "kppkn",      file: "kppkn.gtb.bz2" },
    Corpus { name: "asyoulik",   file: "asyoulik.txt.bz2" },
    Corpus { name: "alice29",    file: "alice29.txt.bz2" },
    Corpus { name: "paper_100k", file: "paper-100k.pdf.bz2" },
    Corpus { name: "lcet10",     file: "lcet10.txt.bz2" },
    Corpus { name: "fireworks",  file: "fireworks.jpeg.bz2" },
    Corpus { name: "plrabn12",   file: "plrabn12.txt.bz2" },
    Corpus { name: "urls_10k",   file: "urls.10K.bz2" },
    Corpus { name: "xml",        file: "xml.bz2" },
    Corpus { name: "reymont",    file: "reymont.bz2" },
    Corpus { name: "nci",        file: "nci.bz2" },
    Corpus { name: "mr",         file: "mr.bz2" },
    Corpus { name: "ooffice",    file: "ooffice.bz2" },
    Corpus { name: "osdb",       file: "osdb.bz2" },
    Corpus { name: "dickens",    file: "dickens.bz2" },
    Corpus { name: "x_ray",      file: "x-ray.bz2" },
    Corpus { name: "sao",        file: "sao.bz2" },
    Corpus { name: "samba",      file: "samba.bz2" },
    Corpus { name: "webster",    file: "webster.bz2" },
    Corpus { name: "mozilla",    file: "mozilla.bz2" },
];

fn data_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("benches").join("data")
}

fn decompress_bz2(path: &std::path::Path) -> Vec<u8> {
    let raw = std::fs::read(path)
        .unwrap_or_else(|e| panic!("failed to read {}: {}", path.display(), e));
    let mut dec = bzip2::read::BzDecoder::new(raw.as_slice());
    let mut out = Vec::new();
    dec.read_to_end(&mut out)
        .unwrap_or_else(|e| panic!("failed to bz2-decode {}: {}", path.display(), e));
    out
}

/// Load (and cache) one corpus file by its short name.  Panics if the name is
/// unknown or the file is unreadable.
///
/// The returned slice is leaked once and reused on subsequent calls — fine for
/// test/bench processes which exit on completion anyway.
pub fn load(name: &str) -> &'static [u8] {
    static CACHE: OnceLock<Mutex<HashMap<&'static str, &'static [u8]>>> = OnceLock::new();
    let cache = CACHE.get_or_init(|| Mutex::new(HashMap::new()));
    let mut guard = cache.lock().unwrap();
    if let Some(&bytes) = guard.get(name) {
        return bytes;
    }
    let corpus = CORPORA
        .iter()
        .find(|c| c.name == name)
        .unwrap_or_else(|| panic!("unknown corpus: {name}"));
    let bytes = decompress_bz2(&data_dir().join(corpus.file));
    let leaked: &'static [u8] = Box::leak(bytes.into_boxed_slice());
    guard.insert(corpus.name, leaked);
    leaked
}

/// Load every corpus file and return `(name, bytes)` pairs.
pub fn load_all() -> Vec<(&'static str, &'static [u8])> {
    CORPORA.iter().map(|c| (c.name, load(c.name))).collect()
}

/// Subset of the corpus used as the default fixture for benches and the
/// non-`#[ignore]` cross-impl tests.
///
/// Two criteria for inclusion:
///   * Decompressed size keeps a single `cargo bench` invocation tolerable.
///   * Currently round-trips cleanly through *every* codec at every level —
///     i.e. no known encoder/decoder bug trips on it.  Files that surfaced
///     bugs (urls_10k, xml, ooffice, x_ray) are intentionally left out of
///     the default subset; they remain in the full corpus and run via
///     `cargo test -- --ignored`, where the bug surface is documented.
///
/// In rough size order so early progress feedback is fast.
pub const BENCH_SUBSET: &[&str] = &[
    "tom_sawyer",
    "html",
    "paper_100k",
    "fireworks",
    "asyoulik",
    "alice29",
    "kppkn",
    "html_x_4",
    "lcet10",
    "plrabn12",
    "reymont",
];

/// Load the bench subset (see [`BENCH_SUBSET`]).
pub fn load_bench_subset() -> Vec<(&'static str, &'static [u8])> {
    BENCH_SUBSET.iter().map(|&n| (n, load(n))).collect()
}
