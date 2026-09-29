
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


Pre-compiled libraries available on [![Anaconda-Server Badge](https://anaconda.org/conda-forge/libcramjam/badges/version.svg)](https://anaconda.org/conda-forge/libcramjam)
