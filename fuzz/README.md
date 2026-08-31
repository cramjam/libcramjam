# Fuzzing

Differential + panic-sweep fuzz targets for the pure-Rust codecs. Each decoder
target decodes arbitrary bytes and, when the C reference also decodes them,
asserts the **lengths** agree (a length divergence = a structural desync bug).
Byte-content divergence on a mutated, checksum-less stream is tolerated GIGO;
valid-stream byte-exactness is covered by `tests/corpus_*.rs` and the
`enc_roundtrip` target (which starts from our own valid encoder output).

Targets: `dec_inflate` (zlib/gzip/deflate), `dec_zstd`, `dec_lz4_frame`,
`dec_lz4_block`, `dec_bzip2`, `dec_xz`, `dec_into` (fixed-buffer panic sweep),
`enc_roundtrip` (encode→decode through ours + C), `enc_stream` (streaming
write/flush patterns).

```bash
cargo +nightly fuzz run dec_xz -- -max_total_time=300
```

Regression inputs for fixed bugs live in `artifacts/<target>/`. The mutation
corpus (`corpus/`) and build output (`target/`) are git-ignored; seed the
corpus with `python3 -c` snippets over `benches/data` (see session notes).
