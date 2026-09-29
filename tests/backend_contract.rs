//! Backend contract: the guarantees that must hold for BOTH the C-backed and
//! the pure-Rust build, checked through the public API only against the
//! reference C crates (dev-dependencies). CI runs this file once per backend,
//! so anything that passes here in both builds means switching backends
//! can't produce data the other side (or the reference libraries) rejects.
//!
//! Covered per codec: every accepted level ours→reference and
//! reference→ours, streaming write/flush/finish, empty input, concatenated
//! streams, truncated and corrupted input (must be `Err`, never wrong data),
//! and a compression-ratio sanity bound against the reference at the same
//! level (catches level-semantics drift, e.g. a level silently storing).

use std::io::{Cursor, Read, Write};

use libcramjam::Backend;

// ---------------------------------------------------------------------------
// Inputs
// ---------------------------------------------------------------------------

fn text(n: usize) -> Vec<u8> {
    const WORDS: &[&str] = &["alpha ", "beta ", "gamma ", "delta ", "lorem ", "ipsum ", "\n", "0123 ", "zeta "];
    let mut s: u64 = 0x9E37_79B9_7F4A_7C15;
    let mut out = Vec::with_capacity(n);
    while out.len() < n {
        s ^= s << 13;
        s ^= s >> 7;
        s ^= s << 17;
        out.extend_from_slice(WORDS[(s % WORDS.len() as u64) as usize].as_bytes());
    }
    out.truncate(n);
    out
}

fn noise(n: usize, seed: u64) -> Vec<u8> {
    let mut s = seed | 1;
    (0..n)
        .map(|_| {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            (s >> 24) as u8
        })
        .collect()
}

/// Inputs every codec is checked on.
fn inputs() -> Vec<(&'static str, Vec<u8>)> {
    let mut mixed = text(150_000);
    mixed.extend(noise(70_000, 3));
    mixed.extend(vec![0u8; 100_000]);
    mixed.extend(text(20_000));
    vec![
        ("empty", vec![]),
        ("one_byte", vec![7]),
        ("text_1k", text(1_000)),
        ("text_300k", text(300_000)),
        ("noise_64k", noise(65_536, 11)),
        ("zeros_1m", vec![0u8; 1 << 20]),
        ("mixed", mixed),
    ]
}

/// Compressible input for the ratio bound.
fn ratio_input() -> Vec<u8> {
    text(400_000)
}

/// Our output may be at most this much larger than the reference's at the
/// same level. Generous: it exists to catch semantic drift (store instead of
/// compress, wrong level table), not to benchmark.
fn assert_ratio_sane(codec: &str, level: impl std::fmt::Debug, ours: usize, reference: usize) {
    assert!(
        ours <= reference + reference / 4 + 64,
        "{codec} level {level:?}: output {ours} B vs reference {reference} B — level semantics diverge from the C library"
    );
}

/// Truncations of a valid stream must never decode to `Ok` with wrong data.
/// `Ok` is only acceptable if the output is exactly the original (e.g. a
/// dropped trailing checksum some decoders don't insist on).
fn assert_truncations_rejected(codec: &str, compressed: &[u8], original: &[u8], decode: impl Fn(&[u8]) -> std::io::Result<Vec<u8>>) {
    let len = compressed.len();
    let mut cuts: Vec<usize> = vec![0, 1, len / 3, len / 2, len - 1];
    cuts.extend((1..=8).filter(|&k| k < len).map(|k| len - k));
    cuts.sort_unstable();
    cuts.dedup();
    for cut in cuts {
        if cut >= len {
            continue;
        }
        if let Ok(out) = decode(&compressed[..cut]) {
            assert!(
                out == original,
                "{codec}: truncation to {cut}/{len} bytes decoded to Ok with {} bytes of wrong data",
                out.len()
            );
        }
    }
}

/// Single-byte corruption of a checksummed stream must be an error (or, if
/// it hits a byte the format ignores, the original data) — never wrong data.
fn assert_corruption_detected(codec: &str, compressed: &[u8], original: &[u8], decode: impl Fn(&[u8]) -> std::io::Result<Vec<u8>>) {
    let len = compressed.len();
    for pos in [len / 4, len / 2, 3 * len / 4, len.saturating_sub(6)] {
        let mut bad = compressed.to_vec();
        bad[pos] ^= 0x55;
        if let Ok(out) = decode(&bad) {
            assert!(out == original, "{codec}: corruption at byte {pos}/{len} decoded to Ok with wrong data");
        }
    }
}

fn expected_backend(pure: bool) -> Backend {
    // CI sets this so a misconfigured job can't silently test the wrong backend.
    match std::env::var("LIBCRAMJAM_EXPECT_BACKEND").as_deref() {
        Ok("c") => Backend::C,
        Ok("pure") => Backend::PureRust,
        Ok(other) => panic!("LIBCRAMJAM_EXPECT_BACKEND must be 'c' or 'pure', got {other:?}"),
        Err(_) if pure => Backend::PureRust,
        Err(_) => Backend::C,
    }
}

#[test]
fn backend_is_the_one_requested() {
    #[cfg(any(feature = "zstd", feature = "zstd-pure"))]
    assert_eq!(libcramjam::zstd::BACKEND, expected_backend(cfg!(feature = "zstd-pure")), "zstd");
    #[cfg(any(feature = "lz4", feature = "lz4-pure"))]
    assert_eq!(libcramjam::lz4::BACKEND, expected_backend(cfg!(feature = "lz4-pure")), "lz4");
    #[cfg(any(feature = "bzip2", feature = "bzip2-pure"))]
    assert_eq!(libcramjam::bzip2::BACKEND, expected_backend(cfg!(feature = "bzip2-pure")), "bzip2");
    #[cfg(any(feature = "xz", feature = "xz-pure"))]
    assert_eq!(libcramjam::xz::BACKEND, expected_backend(cfg!(feature = "xz-pure")), "xz");
    #[cfg(any(feature = "gzip", feature = "deflate-pure"))]
    assert_eq!(libcramjam::gzip::BACKEND, expected_backend(cfg!(feature = "deflate-pure")), "gzip");
    #[cfg(any(feature = "zlib", feature = "deflate-pure"))]
    assert_eq!(libcramjam::zlib::BACKEND, expected_backend(cfg!(feature = "deflate-pure")), "zlib");
    #[cfg(any(feature = "deflate", feature = "deflate-pure"))]
    assert_eq!(libcramjam::deflate::BACKEND, expected_backend(cfg!(feature = "deflate-pure")), "deflate");
}

// ---------------------------------------------------------------------------
// zstd
// ---------------------------------------------------------------------------

#[cfg(any(feature = "zstd", feature = "zstd-pure"))]
mod zstd_contract {
    use super::*;
    use libcramjam::zstd as ours;

    fn compress(data: &[u8], level: i32) -> Vec<u8> {
        let mut out = Vec::new();
        ours::compress(data, &mut out, Some(level), Some(data.len())).unwrap();
        out
    }
    fn decompress(data: &[u8]) -> std::io::Result<Vec<u8>> {
        let mut out = Vec::new();
        ours::decompress(data, &mut out)?;
        Ok(out)
    }
    fn reference_decode(data: &[u8]) -> Vec<u8> {
        zstd::stream::decode_all(data).unwrap()
    }

    #[test]
    fn ours_to_reference_every_level() {
        for (name, data) in inputs() {
            for level in -5..=24 {
                assert_eq!(reference_decode(&compress(&data, level)), data, "zstd level {level} {name}");
            }
        }
    }

    #[test]
    fn reference_to_ours() {
        for (name, data) in inputs() {
            for level in [-3, 1, 3, 9, 19, 22] {
                let c = zstd::stream::encode_all(&data[..], level).unwrap();
                assert_eq!(decompress(&c).unwrap(), data, "zstd reference level {level} {name}");
            }
        }
    }

    #[test]
    fn ratio_matches_reference_semantics() {
        let data = ratio_input();
        for level in [0, 1, 3, 6, 12, 19] {
            let theirs = zstd::stream::encode_all(&data[..], level).unwrap().len();
            assert_ratio_sane("zstd", level, compress(&data, level).len(), theirs);
        }
        // Default (None) is level 3 in both.
        let mut dflt = Vec::new();
        ours::compress(&data[..], &mut dflt, None, None).unwrap();
        assert_ratio_sane("zstd", "default", dflt.len(), zstd::stream::encode_all(&data[..], 3).unwrap().len());
    }

    #[test]
    fn streaming() {
        for (name, data) in inputs() {
            for level in [0, 1, 3, 9] {
                let mut enc = ours::ZstdStreamCompressor::new(Vec::new(), level).unwrap();
                let (a, b) = data.split_at(data.len() / 2);
                enc.write_all(a).unwrap();
                enc.flush().unwrap();
                enc.write_all(b).unwrap();
                enc.flush().unwrap();
                let out = enc.finish().unwrap();
                assert_eq!(reference_decode(&out), data, "zstd stream level {level} {name}");
                assert_eq!(decompress(&out).unwrap(), data, "zstd stream level {level} {name} (ours)");
            }
        }
    }

    #[test]
    fn concatenated_frames() {
        let (a, b) = (text(50_000), noise(10_000, 5));
        let mut both = compress(&a, 3);
        both.extend(zstd::stream::encode_all(&b[..], 3).unwrap());
        assert_eq!(decompress(&both).unwrap(), [a, b].concat());
    }

    #[test]
    fn truncated_and_corrupt() {
        let data = text(200_000);
        let c = compress(&data, 3);
        assert_truncations_rejected("zstd", &c, &data, decompress);
        // zstd frames only carry a checksum when enabled; test with one.
        let mut enc = zstd::stream::Encoder::new(Vec::new(), 3).unwrap();
        enc.include_checksum(true).unwrap();
        enc.write_all(&data).unwrap();
        let checked = enc.finish().unwrap();
        assert_corruption_detected("zstd", &checked, &data, decompress);
    }
}

// ---------------------------------------------------------------------------
// lz4 (frame + block)
// ---------------------------------------------------------------------------

#[cfg(any(feature = "lz4", feature = "lz4-pure"))]
mod lz4_contract {
    use super::*;
    use libcramjam::lz4 as ours;

    fn compress(data: &[u8], level: Option<u32>) -> Vec<u8> {
        let mut out = Vec::new();
        ours::compress(data, &mut out, level).unwrap();
        out
    }
    fn decompress(data: &[u8]) -> std::io::Result<Vec<u8>> {
        let mut out = Vec::new();
        ours::decompress(data, &mut out)?;
        Ok(out)
    }
    fn reference_decode(data: &[u8]) -> Vec<u8> {
        let mut out = Vec::new();
        lz4::Decoder::new(data).unwrap().read_to_end(&mut out).unwrap();
        out
    }
    fn reference_encode(data: &[u8], level: u32, linked: bool, checksum: bool) -> Vec<u8> {
        let mut enc = lz4::EncoderBuilder::new()
            .level(level)
            .block_mode(if linked { lz4::BlockMode::Linked } else { lz4::BlockMode::Independent })
            .checksum(if checksum { lz4::ContentChecksum::ChecksumEnabled } else { lz4::ContentChecksum::NoChecksum })
            .build(Vec::new())
            .unwrap();
        enc.write_all(data).unwrap();
        let (out, r) = enc.finish();
        r.unwrap();
        out
    }

    #[test]
    fn ours_to_reference_every_level() {
        for (name, data) in inputs() {
            for level in (0..=16).map(Some).chain([None]) {
                let c = compress(&data, level);
                assert_eq!(reference_decode(&c), data, "lz4 level {level:?} {name}");
                assert!(c.len() <= ours::compress_bound(data.len(), level), "lz4 bound level {level:?} {name}");
            }
        }
    }

    #[test]
    fn reference_to_ours() {
        for (name, data) in inputs() {
            for level in [0, 1, 4, 9, 12] {
                for (linked, checksum) in [(true, true), (false, true), (true, false), (false, false)] {
                    let c = reference_encode(&data, level, linked, checksum);
                    assert_eq!(decompress(&c).unwrap(), data, "lz4 ref level {level} linked={linked} checksum={checksum} {name}");
                }
            }
        }
    }

    #[test]
    fn ratio_matches_reference_semantics() {
        let data = ratio_input();
        for level in [0, 1, 4, 9, 12] {
            let theirs = reference_encode(&data, level, true, false).len();
            assert_ratio_sane("lz4", level, compress(&data, Some(level)).len(), theirs);
        }
    }

    #[test]
    fn streaming_with_frame_options() {
        for (name, data) in inputs() {
            for level in [0, 4, 9] {
                for (linked, checksum) in [(true, true), (false, true), (true, false), (false, false)] {
                    let mut enc = ours::Lz4StreamCompressor::with_options(Vec::new(), level, linked, checksum);
                    let (a, b) = data.split_at(data.len() / 3);
                    enc.write_all(a).unwrap();
                    enc.flush().unwrap();
                    enc.write_all(b).unwrap();
                    let out = enc.finish().unwrap();
                    let tag = format!("lz4 stream level {level} linked={linked} checksum={checksum} {name}");
                    assert_eq!(reference_decode(&out), data, "{tag}");
                    assert_eq!(decompress(&out).unwrap(), data, "{tag} (ours)");
                }
            }
        }
    }

    #[test]
    fn concatenated_frames() {
        let (a, b) = (text(50_000), noise(10_000, 5));
        let mut both = compress(&a, None);
        both.extend(reference_encode(&b, 9, false, true));
        assert_eq!(decompress(&both).unwrap(), [a, b].concat());
    }

    #[test]
    fn truncated_and_corrupt() {
        let data = text(200_000);
        let c = compress(&data, None);
        assert_truncations_rejected("lz4", &c, &data, decompress);
        let checked = reference_encode(&data, 4, true, true);
        assert_corruption_detected("lz4", &checked, &data, decompress);
    }

    #[test]
    fn block_api_cross() {
        for (name, data) in inputs() {
            // ours → reference
            let ours_block = ours::block::compress_vec(&data, None, None, Some(true)).unwrap();
            assert!(ours_block.len() <= ours::block::compress_bound(data.len(), Some(true)));
            assert_eq!(lz4::block::decompress(&ours_block, None).unwrap(), data, "lz4 block ours→ref {name}");
            let raw = ours::block::compress_vec(&data, None, None, Some(false)).unwrap();
            assert_eq!(lz4::block::decompress(&raw, Some(data.len() as i32)).unwrap(), data, "lz4 raw block ours→ref {name}");
            // compress_into agrees with compress_vec
            let mut buf = vec![0u8; ours::block::compress_bound(data.len(), Some(true))];
            let n = ours::block::compress_into(&data, &mut buf, None, None, Some(true)).unwrap();
            assert_eq!(lz4::block::decompress(&buf[..n], None).unwrap(), data, "lz4 compress_into {name}");
            // reference → ours
            for mode in [lz4::block::CompressionMode::DEFAULT, lz4::block::CompressionMode::HIGHCOMPRESSION(9), lz4::block::CompressionMode::FAST(8)] {
                let theirs = lz4::block::compress(&data, Some(mode), true).unwrap();
                assert_eq!(ours::block::decompress_vec(&theirs).unwrap(), data, "lz4 block ref→ours {name}");
                let mut out = vec![0u8; data.len()];
                let n = ours::block::decompress_into(&theirs, &mut out, Some(true)).unwrap();
                assert_eq!(&out[..n], &data[..], "lz4 block ref→ours decompress_into {name}");
                let theirs_raw = lz4::block::compress(&data, Some(mode), false).unwrap();
                let mut out = vec![0u8; data.len()];
                let n = ours::block::decompress_into(&theirs_raw, &mut out, Some(false)).unwrap();
                assert_eq!(&out[..n], &data[..], "lz4 raw block ref→ours {name}");
            }
        }
    }
}

// ---------------------------------------------------------------------------
// bzip2
// ---------------------------------------------------------------------------

#[cfg(any(feature = "bzip2", feature = "bzip2-pure"))]
mod bzip2_contract {
    use super::*;
    use libcramjam::bzip2 as ours;

    fn compress(data: &[u8], level: Option<u32>) -> Vec<u8> {
        let mut out = Vec::new();
        ours::compress(data, &mut out, level).unwrap();
        out
    }
    fn decompress(data: &[u8]) -> std::io::Result<Vec<u8>> {
        let mut out = Vec::new();
        ours::decompress(data, &mut out)?;
        Ok(out)
    }
    fn reference_decode(data: &[u8]) -> Vec<u8> {
        let mut out = Vec::new();
        bzip2::read::MultiBzDecoder::new(data).read_to_end(&mut out).unwrap();
        out
    }
    fn reference_encode(data: &[u8], level: u32) -> Vec<u8> {
        let mut enc = bzip2::write::BzEncoder::new(Vec::new(), bzip2::Compression::new(level));
        enc.write_all(data).unwrap();
        enc.finish().unwrap()
    }

    #[test]
    fn ours_to_reference_every_level() {
        for (name, data) in inputs() {
            // 0 and >9 are clamped to 1..=9 by both backends.
            for level in (0..=12).map(Some).chain([None]) {
                assert_eq!(reference_decode(&compress(&data, level)), data, "bzip2 level {level:?} {name}");
            }
        }
    }

    #[test]
    fn reference_to_ours() {
        for (name, data) in inputs() {
            for level in [1, 6, 9] {
                assert_eq!(decompress(&reference_encode(&data, level)).unwrap(), data, "bzip2 ref level {level} {name}");
            }
        }
    }

    #[test]
    fn ratio_matches_reference_semantics() {
        let data = ratio_input();
        for level in [1, 6, 9] {
            assert_ratio_sane("bzip2", level, compress(&data, Some(level)).len(), reference_encode(&data, level).len());
        }
    }

    #[test]
    fn streaming() {
        for (name, data) in inputs() {
            for level in [0, 1, 9, 12] {
                let mut enc = ours::Bzip2StreamCompressor::new(Vec::new(), level);
                let (a, b) = data.split_at(data.len() / 2);
                enc.write_all(a).unwrap();
                enc.flush().unwrap();
                enc.write_all(b).unwrap();
                let out = enc.finish().unwrap();
                assert_eq!(reference_decode(&out), data, "bzip2 stream level {level} {name}");
                assert_eq!(decompress(&out).unwrap(), data, "bzip2 stream level {level} {name} (ours)");
            }
        }
    }

    #[test]
    fn concatenated_streams() {
        let (a, b) = (text(50_000), noise(10_000, 5));
        let mut both = compress(&a, None);
        both.extend(reference_encode(&b, 9));
        assert_eq!(decompress(&both).unwrap(), [a, b].concat());
    }

    #[test]
    fn truncated_and_corrupt() {
        let data = text(200_000);
        let c = compress(&data, None);
        assert_truncations_rejected("bzip2", &c, &data, decompress);
        assert_corruption_detected("bzip2", &c, &data, decompress);
    }
}

// ---------------------------------------------------------------------------
// deflate / gzip / zlib
// ---------------------------------------------------------------------------

#[cfg(any(all(feature = "deflate", feature = "gzip", feature = "zlib"), feature = "deflate-pure"))]
mod deflate_family_contract {
    use super::*;

    #[derive(Clone, Copy, Debug)]
    enum Kind {
        Deflate,
        Gzip,
        Zlib,
    }
    const KINDS: [Kind; 3] = [Kind::Deflate, Kind::Gzip, Kind::Zlib];

    fn compress(kind: Kind, data: &[u8], level: Option<u32>) -> Vec<u8> {
        let mut out = Vec::new();
        match kind {
            Kind::Deflate => libcramjam::deflate::compress(data, &mut out, level),
            Kind::Gzip => libcramjam::gzip::compress(data, &mut out, level),
            Kind::Zlib => libcramjam::zlib::compress(data, &mut out, level),
        }
        .unwrap();
        out
    }
    fn decompress(kind: Kind, data: &[u8]) -> std::io::Result<Vec<u8>> {
        let mut out = Vec::new();
        match kind {
            Kind::Deflate => libcramjam::deflate::decompress(data, &mut out),
            Kind::Gzip => libcramjam::gzip::decompress(data, &mut out),
            Kind::Zlib => libcramjam::zlib::decompress(data, &mut out),
        }?;
        Ok(out)
    }
    fn bound(kind: Kind, n: usize) -> usize {
        match kind {
            Kind::Deflate => libcramjam::deflate::compress_bound(n),
            Kind::Gzip => libcramjam::gzip::compress_bound(n),
            Kind::Zlib => libcramjam::zlib::compress_bound(n),
        }
    }
    fn reference_decode(kind: Kind, data: &[u8]) -> Vec<u8> {
        let mut out = Vec::new();
        match kind {
            Kind::Deflate => flate2::read::DeflateDecoder::new(data).read_to_end(&mut out),
            Kind::Gzip => flate2::read::MultiGzDecoder::new(data).read_to_end(&mut out),
            Kind::Zlib => flate2::read::ZlibDecoder::new(data).read_to_end(&mut out),
        }
        .unwrap();
        out
    }
    fn reference_encode(kind: Kind, data: &[u8], level: u32) -> Vec<u8> {
        let c = flate2::Compression::new(level);
        match kind {
            Kind::Deflate => {
                let mut e = flate2::write::DeflateEncoder::new(Vec::new(), c);
                e.write_all(data).unwrap();
                e.finish().unwrap()
            }
            Kind::Gzip => {
                let mut e = flate2::write::GzEncoder::new(Vec::new(), c);
                e.write_all(data).unwrap();
                e.finish().unwrap()
            }
            Kind::Zlib => {
                let mut e = flate2::write::ZlibEncoder::new(Vec::new(), c);
                e.write_all(data).unwrap();
                e.finish().unwrap()
            }
        }
    }

    #[test]
    fn ours_to_reference_every_level() {
        for kind in KINDS {
            for (name, data) in inputs() {
                for level in (0..=12).map(Some).chain([None]) {
                    let c = compress(kind, &data, level);
                    assert_eq!(reference_decode(kind, &c), data, "{kind:?} level {level:?} {name}");
                    assert!(c.len() <= bound(kind, data.len()), "{kind:?} bound level {level:?} {name}");
                }
            }
        }
    }

    #[test]
    fn reference_to_ours() {
        for kind in KINDS {
            for (name, data) in inputs() {
                for level in [0, 1, 6, 9] {
                    let c = reference_encode(kind, &data, level);
                    assert_eq!(decompress(kind, &c).unwrap(), data, "{kind:?} ref level {level} {name}");
                }
            }
        }
    }

    #[test]
    fn ratio_matches_reference_semantics() {
        let data = ratio_input();
        for kind in KINDS {
            for level in [0, 1, 6, 9] {
                let theirs = reference_encode(kind, &data, level).len();
                assert_ratio_sane(&format!("{kind:?}"), level, compress(kind, &data, Some(level)).len(), theirs);
            }
        }
    }

    #[test]
    fn streaming() {
        for (name, data) in inputs() {
            for level in [0, 1, 6, 9] {
                macro_rules! drive {
                    ($kind:expr, $enc:expr) => {{
                        let mut enc = $enc;
                        let (a, b) = data.split_at(data.len() / 2);
                        enc.write_all(a).unwrap();
                        enc.flush().unwrap();
                        enc.flush().unwrap();
                        enc.write_all(b).unwrap();
                        let out = enc.finish().unwrap();
                        assert_eq!(reference_decode($kind, &out), data, "{:?} stream level {level} {name}", $kind);
                        assert_eq!(decompress($kind, &out).unwrap(), data, "{:?} stream level {level} {name} (ours)", $kind);
                    }};
                }
                drive!(Kind::Deflate, libcramjam::deflate::DeflateStreamCompressor::new(Vec::new(), level));
                drive!(Kind::Gzip, libcramjam::gzip::GzipStreamCompressor::new(Vec::new(), level));
                drive!(Kind::Zlib, libcramjam::zlib::ZlibStreamCompressor::new(Vec::new(), level));
            }
        }
    }

    #[test]
    fn gzip_concatenated_members() {
        let (a, b) = (text(50_000), noise(10_000, 5));
        let mut both = compress(Kind::Gzip, &a, None);
        both.extend(reference_encode(Kind::Gzip, &b, 9));
        assert_eq!(decompress(Kind::Gzip, &both).unwrap(), [a, b].concat());
    }

    #[test]
    fn truncated_and_corrupt() {
        let data = text(200_000);
        for kind in KINDS {
            let c = compress(kind, &data, None);
            assert_truncations_rejected(&format!("{kind:?}"), &c, &data, |d| decompress(kind, d));
        }
        // Raw deflate has no checksum; gzip (CRC32) and zlib (Adler-32) do.
        for kind in [Kind::Gzip, Kind::Zlib] {
            let c = compress(kind, &data, None);
            assert_corruption_detected(&format!("{kind:?}"), &c, &data, |d| decompress(kind, d));
        }
    }
}

// ---------------------------------------------------------------------------
// xz / lzma
// ---------------------------------------------------------------------------

#[cfg(any(feature = "xz", feature = "xz-pure"))]
mod xz_contract {
    use super::*;
    use libcramjam::xz::{self as ours, Check, Filters, Format, LzmaOptions};

    fn compress(data: &[u8], preset: Option<u32>, format: Option<Format>, check: Option<Check>, filters: Option<Filters>, options: Option<LzmaOptions>) -> std::io::Result<Vec<u8>> {
        let mut out = Vec::new();
        ours::compress(data, &mut out, preset, format, check, filters, options)?;
        Ok(out)
    }
    fn xz(data: &[u8], preset: u32) -> Vec<u8> {
        compress(data, Some(preset), None, None, None, None).unwrap()
    }
    fn decompress(data: &[u8]) -> std::io::Result<Vec<u8>> {
        let mut out = Vec::new();
        ours::decompress(data, &mut out)?;
        Ok(out)
    }
    fn reference_decode_xz(data: &[u8]) -> Vec<u8> {
        let mut out = Vec::new();
        xz2::read::XzDecoder::new_multi_decoder(data).read_to_end(&mut out).unwrap();
        out
    }
    fn reference_decode_alone(data: &[u8]) -> Vec<u8> {
        let stream = xz2::stream::Stream::new_lzma_decoder(u64::MAX).unwrap();
        let mut out = Vec::new();
        xz2::read::XzDecoder::new_stream(data, stream).read_to_end(&mut out).unwrap();
        out
    }
    fn reference_encode(data: &[u8], preset: u32) -> Vec<u8> {
        let mut e = xz2::write::XzEncoder::new(Vec::new(), preset);
        e.write_all(data).unwrap();
        e.finish().unwrap()
    }

    #[test]
    fn ours_to_reference_every_preset() {
        for (name, data) in inputs() {
            for preset in (0..=9).chain([6 | 0x8000_0000]) {
                assert_eq!(reference_decode_xz(&xz(&data, preset)), data, "xz preset {preset:#x} {name}");
            }
        }
    }

    #[test]
    fn invalid_presets_rejected() {
        for preset in [10, 31, 0x4000_0006] {
            assert!(compress(b"abc", Some(preset), None, None, None, None).is_err(), "preset {preset:#x}");
            let mut enc = ours::XzStreamCompressor::new(Vec::new(), preset);
            assert!(enc.write_all(b"abc").is_err(), "stream preset {preset:#x}");
        }
    }

    #[test]
    fn reference_to_ours() {
        for (name, data) in inputs() {
            for preset in [0, 1, 6, 9] {
                assert_eq!(decompress(&reference_encode(&data, preset)).unwrap(), data, "xz ref preset {preset} {name}");
            }
        }
    }

    #[test]
    fn ratio_matches_reference_semantics() {
        let data = ratio_input();
        for preset in [0, 3, 6, 9] {
            assert_ratio_sane("xz", preset, xz(&data, preset).len(), reference_encode(&data, preset).len());
        }
    }

    #[test]
    fn every_check_decodes_with_liblzma() {
        let data = text(100_000);
        for check in [Check::Crc64, Check::Crc32, Check::Sha256, Check::None] {
            let c = compress(&data, None, Some(Format::XZ), Some(check), None, None).unwrap();
            assert_eq!(reference_decode_xz(&c), data, "xz check {check:?}");
            assert_eq!(decompress(&c).unwrap(), data, "xz check {check:?} (ours)");
        }
    }

    #[test]
    fn filters_and_options_are_honored_in_xz_format() {
        let data = text(200_000);
        let mut opts = LzmaOptions::new_preset(6).unwrap();
        opts.dict_size(1 << 16).literal_context_bits(0).literal_position_bits(2).position_bits(0);
        let mut chain = Filters::new();
        chain.x86().lzma2(&opts);
        for (label, filters, options) in [
            ("bcj+lzma2", Some(chain.clone()), None),
            ("options only", None, Some(opts.clone())),
        ] {
            let c = compress(&data, None, Some(Format::XZ), None, filters, options).unwrap();
            assert_eq!(reference_decode_xz(&c), data, "xz {label}");
            assert_eq!(decompress(&c).unwrap(), data, "xz {label} (ours)");
            // A custom chain must actually be used, not silently replaced by
            // the preset: a 64 KiB dictionary on 200 KB of text differs from
            // preset 6's 8 MiB one.
            assert_ne!(c, xz(&data, 6), "xz {label}: filters/options were ignored");
        }
    }

    #[test]
    fn alone_format() {
        for (name, data) in inputs() {
            let c = compress(&data, Some(6), Some(Format::ALONE), None, None, None).unwrap();
            assert_eq!(reference_decode_alone(&c), data, "lzma alone {name}");
            assert_eq!(decompress(&c).unwrap(), data, "lzma alone {name} (ours)");
        }
        let mut bcj = Filters::new();
        bcj.x86().lzma1(&LzmaOptions::new_preset(6).unwrap());
        assert!(compress(b"abc", None, Some(Format::ALONE), None, Some(bcj), None).is_err());
    }

    #[test]
    fn raw_format_is_container_less_and_round_trips() {
        for (name, data) in inputs() {
            for lzma1 in [false, true] {
                let opts = LzmaOptions::new_preset(6).unwrap();
                let mut f = Filters::new();
                if lzma1 {
                    f.lzma1(&opts);
                } else {
                    f.lzma2(&opts);
                }
                let c = compress(&data, None, Some(Format::RAW), None, Some(f.clone()), None).unwrap();
                assert!(!c.starts_with(b"\xfd7zXZ\x00"), "RAW must not be an .xz container");
                let mut out = Vec::new();
                ours::decompress_raw(&c[..], &mut out, f).unwrap();
                assert_eq!(out, data, "xz raw lzma1={lzma1} {name}");
            }
        }
    }

    #[test]
    fn streaming() {
        for (name, data) in inputs() {
            for preset in [0, 6] {
                let mut enc = ours::XzStreamCompressor::new(Vec::new(), preset);
                let (a, b) = data.split_at(data.len() / 2);
                enc.write_all(a).unwrap();
                enc.flush().unwrap();
                let flushed = enc.get_ref().len();
                enc.write_all(b).unwrap();
                let out = enc.finish().unwrap();
                assert!(flushed > 0);
                assert_eq!(reference_decode_xz(&out), data, "xz stream preset {preset} {name}");
                assert_eq!(decompress(&out).unwrap(), data, "xz stream preset {preset} {name} (ours)");
            }
        }
    }

    #[test]
    fn stream_decompressor_reads_everything() {
        let data = text(100_000);
        let mut out = Vec::new();
        ours::XzStreamDecompressor::new(&xz(&data, 6)[..]).read_to_end(&mut out).unwrap();
        assert_eq!(out, data);
    }

    #[test]
    fn concatenated_streams() {
        let (a, b) = (text(50_000), noise(10_000, 5));
        let mut both = xz(&a, 6);
        both.extend(reference_encode(&b, 1));
        assert_eq!(decompress(&both).unwrap(), [a, b].concat());
    }

    #[test]
    fn truncated_and_corrupt() {
        let data = text(200_000);
        let c = xz(&data, 6);
        assert_truncations_rejected("xz", &c, &data, decompress);
        assert_corruption_detected("xz", &c, &data, decompress);
    }
}

#[test]
fn stream_types_are_send() {
    // The Python bindings keep compressors inside `#[pyclass]` objects.
    fn send<T: Send>() {}
    #[cfg(any(feature = "zstd", feature = "zstd-pure"))]
    send::<libcramjam::zstd::ZstdStreamCompressor<Cursor<Vec<u8>>>>();
    #[cfg(any(feature = "lz4", feature = "lz4-pure"))]
    send::<libcramjam::lz4::Lz4StreamCompressor<Cursor<Vec<u8>>>>();
    #[cfg(any(feature = "bzip2", feature = "bzip2-pure"))]
    send::<libcramjam::bzip2::Bzip2StreamCompressor<Cursor<Vec<u8>>>>();
    #[cfg(any(feature = "xz", feature = "xz-pure"))]
    send::<libcramjam::xz::XzStreamCompressor<Cursor<Vec<u8>>>>();
    #[cfg(any(feature = "gzip", feature = "deflate-pure"))]
    send::<libcramjam::gzip::GzipStreamCompressor<Cursor<Vec<u8>>>>();
    #[cfg(any(feature = "zlib", feature = "deflate-pure"))]
    send::<libcramjam::zlib::ZlibStreamCompressor<Cursor<Vec<u8>>>>();
    #[cfg(any(feature = "deflate", feature = "deflate-pure"))]
    send::<libcramjam::deflate::DeflateStreamCompressor<Cursor<Vec<u8>>>>();
}

/// Bytes after the last stream never show up in the output, and both
/// backends agree on accept/reject, following the C libraries: raw deflate
/// and zlib stop at the end of the stream (their framing has no notion of a
/// next member); xz accepts zero stream padding in multiples of 4 bytes;
/// everything else must be a following stream or it's an error.
#[test]
fn trailing_garbage_never_decoded() {
    let data = text(20_000);
    let mut cases: Vec<(&str, Vec<u8>, Box<dyn Fn(&[u8]) -> std::io::Result<Vec<u8>>>)> = Vec::new();
    macro_rules! case {
        ($name:expr, $compressed:expr, $decompress:path) => {
            cases.push(($name, $compressed, Box::new(|d: &[u8]| {
                let mut out = Vec::new();
                $decompress(d, &mut out).map(|_| out)
            })));
        };
    }
    let mut c = Vec::new();
    #[cfg(any(feature = "zstd", feature = "zstd-pure"))]
    {
        libcramjam::zstd::compress(&data[..], &mut c, None, None).unwrap();
        case!("zstd", std::mem::take(&mut c), libcramjam::zstd::decompress);
    }
    #[cfg(any(feature = "lz4", feature = "lz4-pure"))]
    {
        libcramjam::lz4::compress(&data[..], &mut c, None).unwrap();
        case!("lz4", std::mem::take(&mut c), libcramjam::lz4::decompress);
    }
    #[cfg(any(feature = "bzip2", feature = "bzip2-pure"))]
    {
        libcramjam::bzip2::compress(&data[..], &mut c, None).unwrap();
        case!("bzip2", std::mem::take(&mut c), libcramjam::bzip2::decompress);
    }
    #[cfg(any(all(feature = "deflate", feature = "gzip", feature = "zlib"), feature = "deflate-pure"))]
    {
        libcramjam::gzip::compress(&data[..], &mut c, None).unwrap();
        case!("gzip", std::mem::take(&mut c), libcramjam::gzip::decompress);
        libcramjam::zlib::compress(&data[..], &mut c, None).unwrap();
        case!("zlib", std::mem::take(&mut c), libcramjam::zlib::decompress);
        libcramjam::deflate::compress(&data[..], &mut c, None).unwrap();
        case!("deflate", std::mem::take(&mut c), libcramjam::deflate::decompress);
    }
    #[cfg(any(feature = "xz", feature = "xz-pure"))]
    {
        libcramjam::xz::compress(&data[..], &mut c, None, None::<libcramjam::xz::Format>, None::<libcramjam::xz::Check>, None::<libcramjam::xz::Filters>, None::<libcramjam::xz::LzmaOptions>).unwrap();
        case!("xz", std::mem::take(&mut c), libcramjam::xz::decompress);
    }
    for (name, compressed, decompress) in &cases {
        for garbage in [&b"\0"[..], b"\0\0\0\0", b"garbage!", b"BZh9garbage", b"\x28\xb5\x2f\xfd"] {
            let mut input = compressed.clone();
            input.extend_from_slice(garbage);
            let accepted = matches!(*name, "zlib" | "deflate") || (*name == "xz" && garbage == b"\0\0\0\0");
            match decompress(&input) {
                Ok(out) => {
                    assert!(out == data, "{name}: trailing {garbage:?} leaked into the output");
                    assert!(accepted, "{name}: trailing {garbage:?} must be an error");
                }
                Err(e) => assert!(!accepted, "{name}: trailing {garbage:?} must be ignored, got {e}"),
            }
        }
    }
}
