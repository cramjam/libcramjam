//! Streaming deflate / gzip / zlib compressors: `flush` must make every
//! byte written so far decodable (zlib `Z_SYNC_FLUSH`), `finish` ends the
//! stream; the C reference decoders (flate2) must accept everything.

use std::io::{Read, Write};

use libcramjam::deflate::DeflateStreamCompressor;
use libcramjam::gzip::GzipStreamCompressor;
use libcramjam::zlib::ZlibStreamCompressor;

fn text(n: usize) -> Vec<u8> {
    b"streaming deflate: sync flush, then more data, then finish. ".iter().cycle().take(n).copied().collect()
}

fn noise(n: usize, seed: u32) -> Vec<u8> {
    let mut x = seed | 1;
    (0..n)
        .map(|_| {
            x ^= x << 13;
            x ^= x >> 17;
            x ^= x << 5;
            (x >> 24) as u8
        })
        .collect()
}

#[derive(Clone, Copy)]
enum Kind {
    Deflate,
    Gzip,
    Zlib,
}

/// Decode with flate2 as far as the (possibly unterminated) stream allows.
fn c_decode_prefix(kind: Kind, data: &[u8]) -> Vec<u8> {
    let mut out = Vec::new();
    let mut buf = [0u8; 8192];
    macro_rules! drain {
        ($dec:expr) => {{
            let mut dec = $dec;
            loop {
                match dec.read(&mut buf) {
                    Ok(0) => break,
                    Ok(n) => out.extend_from_slice(&buf[..n]),
                    Err(e) if e.kind() == std::io::ErrorKind::UnexpectedEof => break,
                    Err(e) => panic!("flate2 decode error: {e}"),
                }
            }
        }};
    }
    match kind {
        Kind::Deflate => drain!(flate2::read::DeflateDecoder::new(data)),
        Kind::Gzip => drain!(flate2::read::GzDecoder::new(data)),
        Kind::Zlib => drain!(flate2::read::ZlibDecoder::new(data)),
    }
    out
}

/// Decode a complete stream with flate2 (must succeed and hit EOF cleanly).
fn c_decode(kind: Kind, data: &[u8]) -> Vec<u8> {
    let mut out = Vec::new();
    match kind {
        Kind::Deflate => flate2::read::DeflateDecoder::new(data).read_to_end(&mut out).unwrap(),
        Kind::Gzip => flate2::read::GzDecoder::new(data).read_to_end(&mut out).unwrap(),
        Kind::Zlib => flate2::read::ZlibDecoder::new(data).read_to_end(&mut out).unwrap(),
    };
    out
}

/// Also decode with our own inflater through the public API.
fn ours_decode(kind: Kind, data: &[u8]) -> Vec<u8> {
    let mut out = Vec::new();
    let mut cur = std::io::Cursor::new(data);
    match kind {
        Kind::Deflate => libcramjam::deflate::decompress(&mut cur, &mut out).unwrap(),
        Kind::Gzip => libcramjam::gzip::decompress(&mut cur, &mut out).unwrap(),
        Kind::Zlib => libcramjam::zlib::decompress(&mut cur, &mut out).unwrap(),
    };
    out
}

/// Runs the write/flush/write/flush/write/finish protocol through a
/// compressor generic over `Kind`, checking after every flush.
fn run(kind: Kind, level: u32, parts: &[&[u8]]) {
    let out: Vec<u8> = Vec::new();
    let mut expected: Vec<u8> = Vec::new();
    let mut prefixes: Vec<(usize, usize)> = Vec::new(); // (output len after flush, input len)

    macro_rules! drive {
        ($enc:expr) => {{
            let mut enc = $enc;
            for (i, part) in parts.iter().enumerate() {
                enc.write_all(part).unwrap();
                expected.extend_from_slice(part);
                if i + 1 < parts.len() {
                    enc.flush().unwrap();
                    prefixes.push((enc.get_ref().len(), expected.len()));
                    // A second flush with nothing new: the pure-Rust backend
                    // emits nothing; zlib/flate2 emit another (valid) empty
                    // sync-flush block, which the decodes below cover.
                    enc.flush().unwrap();
                    if libcramjam::deflate::BACKEND == libcramjam::Backend::PureRust {
                        assert_eq!(enc.get_ref().len(), prefixes.last().unwrap().0, "empty flush emitted bytes");
                    }
                }
            }
            enc.finish().unwrap()
        }};
    }
    let full: Vec<u8> = match kind {
        Kind::Deflate => drive!(DeflateStreamCompressor::new(out, level)),
        Kind::Gzip => drive!(GzipStreamCompressor::new(out, level)),
        Kind::Zlib => drive!(ZlibStreamCompressor::new(out, level)),
    };

    assert_eq!(c_decode(kind, &full), expected, "level {level}: flate2 full decode");
    assert_eq!(ours_decode(kind, &full), expected, "level {level}: our full decode");
    for &(out_len, in_len) in &prefixes {
        let got = c_decode_prefix(kind, &full[..out_len]);
        assert_eq!(got, &expected[..in_len], "level {level}: flushed prefix must decode to everything written");
    }
}

#[test]
fn flush_makes_written_data_decodable() {
    let a = text(70_000);
    let b = noise(150_000, 7);
    let c = text(10_000);
    for kind in [Kind::Deflate, Kind::Gzip, Kind::Zlib] {
        for level in [0u32, 1, 3, 6, 9] {
            run(kind, level, &[&a, &b, &c]);
        }
    }
}

#[test]
fn small_and_empty_streams() {
    for kind in [Kind::Deflate, Kind::Gzip, Kind::Zlib] {
        for level in [0u32, 1, 6, 9] {
            // Nothing written at all.
            run(kind, level, &[]);
            // Single 1-byte write.
            run(kind, level, &[b"x"]);
            // Flush before any data: an empty first part.
            run(kind, level, &[b"", b"hello"]);
            // Tiny parts with flushes in between.
            run(kind, level, &[b"a", b"b", b"c"]);
        }
    }
}

#[test]
fn many_windows_multi_megabyte() {
    // Spans dozens of 64 KiB windows so the buffer slides many times;
    // repeats farther back than 32 KiB must not be referenced.
    let mut data = Vec::new();
    for i in 0..48u32 {
        data.extend_from_slice(&text(40_000));
        data.extend_from_slice(&noise(30_000, i + 3));
    }
    assert!(data.len() > 3_000_000);
    let mid = data.len() / 3;
    for kind in [Kind::Deflate, Kind::Gzip, Kind::Zlib] {
        for level in [1u32, 6, 9] {
            run(kind, level, &[&data[..mid], &data[mid..2 * mid], &data[2 * mid..]]);
        }
    }
    // Many small writes with flushes: each write ends up decodable.
    let parts: Vec<&[u8]> = data.chunks(97_001).collect();
    run(Kind::Zlib, 6, &parts);
}

#[test]
fn streaming_matches_one_shot_bytes_when_finished_in_one_go() {
    // One write + finish takes the same decisions as the one-shot encoder.
    let data = {
        let mut d = text(300_000);
        d.extend_from_slice(&noise(100_000, 11));
        d
    };
    for level in [1u32, 4, 6, 9] {
        let mut one_shot = Vec::new();
        libcramjam::deflate::compress(&mut std::io::Cursor::new(&data), &mut one_shot, Some(level)).unwrap();
        let mut enc = DeflateStreamCompressor::new(Vec::new(), level);
        enc.write_all(&data).unwrap();
        let streamed = enc.finish().unwrap();
        assert_eq!(streamed, one_shot, "level {level}");
    }
}

#[test]
fn memory_stays_bounded() {
    // 64 MiB written in 1 MiB pieces through the raw stream compressor:
    // the compressor must not retain the input (the Vec would be 64 MiB).
    // We can't measure the private buffer directly, so drive a lot of data
    // and rely on the slide logic via the decode check; keep the amount
    // large enough that a retained-input bug would be obvious in RSS.
    let piece = noise(1 << 20, 99);
    let mut enc = ZlibStreamCompressor::new(Vec::new(), 1);
    let mut expected_len = 0usize;
    for _ in 0..64 {
        enc.write_all(&piece).unwrap();
        expected_len += piece.len();
    }
    let out = enc.finish().unwrap();
    assert_eq!(c_decode(Kind::Zlib, &out).len(), expected_len);
}
