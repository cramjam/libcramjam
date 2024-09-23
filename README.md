
# cramjam library

A Rust library combining different compression algorithms/libraries in a common (as possible) API.


[![CI](https://github.com/cramjam/libcramjam/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/cramjam/libcramjam/actions/workflows/CI.yml)
[![Latest version](https://img.shields.io/crates/v/libcramjam.svg)](https://crates.io/crates/libcramjam)
[![Documentation](https://docs.rs/libcramjam/badge.svg)](https://docs.rs/libcramjam)
![License](https://img.shields.io/crates/l/libcramjam.svg)

Features:

- `snappy`
- `lz4`
- `bzip2`
- `brotli`
- `zstd`
- `igzip`
- `xz`
  - `xz-static`
  - `xz-shared`
- `gzip`
  - `gzip-static`
  - `gzip-shared`
- `deflate`
  - `deflate-static`
  - `deflate-shared`
- `blosc2`
  - `blosc2-shared`
  - `blosc2-static`
- `capi`: Build a C-ABI library. Compatible with [`cargo-c`](https://github.com/lu-zero/cargo-c)


Pre-compiled libraries available on [![Anaconda-Server Badge](https://anaconda.org/conda-forge/libcramjam/badges/version.svg)](https://anaconda.org/conda-forge/libcramjam)
