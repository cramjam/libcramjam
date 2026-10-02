
# cramjam library

A Rust library combining different compression algorithms/libraries in a common (as possible) API.


[![CI](https://github.com/cramjam/libcramjam/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/cramjam/libcramjam/actions/workflows/CI.yml)
[![Latest version](https://img.shields.io/crates/v/libcramjam.svg)](https://crates.io/crates/libcramjam)
[![Documentation](https://docs.rs/libcramjam/badge.svg)](https://docs.rs/libcramjam)
![License](https://img.shields.io/crates/l/libcramjam.svg)

---

#### Features 

(dynamic/static build features available on some variants, check [Cargo.toml](./Cargo.toml)):

- `snappy`
- `lz4`
- `bzip2`
- `brotli`
- `zstd`
- `zlib`
- `xz`
- `gzip`
- `deflate`
- `blosc2`
- `igzip`  (GZIP using ISA-L backend)
- `ideflate`  (DEFLATE using ISA-L backend)
- `izlib`  (ZLIB using ISA-L backend)
- `capi`: Build a C-ABI library. Compatible with [`cargo-c`](https://github.com/lu-zero/cargo-c)

#### Pure-Rust backends (beta)

`zstd`, `lz4`, `bzip2`, `xz` and `deflate`/`gzip`/`zlib` also have pure-Rust
implementations, selected per codec with `zstd-pure`, `lz4-pure`,
`bzip2-pure`, `xz-pure` and `deflate-pure` (covers deflate, gzip and zlib), or
all at once with `pure-rust`. The public API is identical either way; the
C-backed implementations remain the default.

For a build with no C codec libraries at all:

```toml
libcramjam = { version = "0.9", default-features = false, features = ["pure-rust", "snappy", "brotli"] }
```

If both a codec feature and its `-pure` feature end up enabled (Cargo unifies
features across the dependency graph), the pure-Rust backend is used. Each
codec module exposes `BACKEND` so callers can check which one they got.

Both backends run the same test suite, including `tests/backend_contract.rs`,
which checks each against the reference C libraries. The test suite runs once
per backend: `cargo test --release --tests` (C) and
`cargo test --release --tests --no-default-features --features pure-rust,snappy,brotli`.
The benches (`cargo bench --features pure-rust`) compare the pure-Rust
backend with the C libraries.

Measured 2026-10-02 on one x86-64 Linux workstation over the bench subset
(11 files, 8.3 MB), default levels, median of 5, ours and C interleaved per
iteration (`cargo bench --bench summary --features pure-rust`). Compressed
output is byte-identical to the C libraries for zstd, lz4, bzip2 and xz, and
equal in size for deflate at levels 4 to 9 (smaller at levels 1 to 3).

| codec          | compress vs C | decompress vs C |
|----------------|---------------|-----------------|
| deflate / gzip | 1.00x         | 1.6x faster     |
| zstd           | 1.04x faster  | 1.08x faster    |
| lz4            | 1.05x faster  | 1.9x faster     |
| bzip2          | 1.02x slower  | 1.3x faster     |
| xz             | 1.2x faster   | 1.2x faster     |

Ratios swing a few percent run to run; treat anything within 5% as parity.

The pure-Rust codecs keep `unsafe` only in encoder hot loops, SIMD and asm
kernels, and output-buffer cursors. Every block carries a `SAFETY` comment
(enforced by `clippy::undocumented_unsafe_blocks`), and the unit tests run
under Miri in CI.


Pre-compiled libraries available on [![Anaconda-Server Badge](https://anaconda.org/conda-forge/libcramjam/badges/version.svg)](https://anaconda.org/conda-forge/libcramjam)
