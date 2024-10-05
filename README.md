
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


Pre-compiled libraries available on [![Anaconda-Server Badge](https://anaconda.org/conda-forge/libcramjam/badges/version.svg)](https://anaconda.org/conda-forge/libcramjam)
