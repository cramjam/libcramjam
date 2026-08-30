//! Streaming zstd compressor semantics (`ZSTD_compressStream2`):
//! `write` feeds data, `flush` makes everything written so far decodable,
//! `finish` ends the frame. Verified against the C `zstd` crate's decoder.

use std::io::{Cursor, Read, Write};

use libcramjam::zstd::ZstdStreamCompressor;

fn c_decode_all(data: &[u8]) -> Vec<u8> {
    let mut dec = zstd::stream::read::Decoder::new(data).unwrap();
    let mut out = Vec::new();
    dec.read_to_end(&mut out).unwrap();
    out
}

/// Decode a *prefix* of a frame with the C streaming decoder: read until it
/// runs out of input (short read or error at the truncated end).
fn c_decode_prefix(data: &[u8]) -> Vec<u8> {
    let mut dec = zstd::stream::read::Decoder::new(data).unwrap();
    let mut out = Vec::new();
    let mut chunk = [0u8; 65536];
    loop {
        match dec.read(&mut chunk) {
            Ok(0) => break,
            Ok(n) => out.extend_from_slice(&chunk[..n]),
            Err(_) => break,
        }
    }
    out
}

fn ours_decode(data: &[u8]) -> Vec<u8> {
    let mut out = Vec::new();
    libcramjam::zstd::decompress(&mut Cursor::new(data), &mut out).unwrap();
    out
}

fn text(n: usize, seed: u32) -> Vec<u8> {
    let words = ["the ", "quick ", "brown ", "fox ", "jumps ", "over ", "lazy ", "dogs ", "\n", "zstd ", "stream "];
    let mut v = Vec::with_capacity(n + 16);
    let mut x = seed.wrapping_mul(2654435761) | 1;
    while v.len() < n {
        x ^= x << 13;
        x ^= x >> 17;
        x ^= x << 5;
        v.extend_from_slice(words[(x % words.len() as u32) as usize].as_bytes());
    }
    v.truncate(n);
    v
}

fn compressor(level: i32) -> ZstdStreamCompressor<Cursor<Vec<u8>>> {
    ZstdStreamCompressor::new(Cursor::new(Vec::new()), level).unwrap()
}

/// Bytes the compressor has emitted so far.
fn emitted(c: &ZstdStreamCompressor<Cursor<Vec<u8>>>) -> Vec<u8> {
    c.get_ref().get_ref().clone()
}

#[test]
fn flush_makes_written_data_decodable() {
    for level in [1i32, 3, 9] {
        let a = text(300_000, 1);
        let b = text(70_000, 2);
        let c = text(5, 3);
        let mut comp = compressor(level);

        comp.write_all(&a).unwrap();
        comp.flush().unwrap();
        let after_a = emitted(&comp);
        assert_eq!(c_decode_prefix(&after_a), a, "level {level}: prefix after first flush");
        assert_eq!(ours_decode_prefix(&after_a), a);

        comp.write_all(&b).unwrap();
        comp.flush().unwrap();
        let after_b = emitted(&comp);
        assert_eq!(c_decode_prefix(&after_b), [a.clone(), b.clone()].concat(), "level {level}: prefix after second flush");

        comp.write_all(&c).unwrap();
        let whole = comp.finish().unwrap().into_inner();
        let expected = [a, b, c].concat();
        assert_eq!(c_decode_all(&whole), expected, "level {level}");
        assert_eq!(ours_decode(&whole), expected, "level {level}");
    }
}

/// Our decoder on a truncated frame: decode what is there.
fn ours_decode_prefix(data: &[u8]) -> Vec<u8> {
    // The pure-Rust decoder needs a complete frame; append an empty last
    // block to close the frame we have so far (the prefix ends on a block
    // boundary after a flush).
    let mut closed = data.to_vec();
    closed.extend_from_slice(&[0x01, 0x00, 0x00]);
    ours_decode(&closed)
}

#[test]
fn empty_and_tiny_inputs() {
    for level in [1i32, 3, 9] {
        // Nothing written at all.
        let out = compressor(level).finish().unwrap().into_inner();
        assert!(c_decode_all(&out).is_empty(), "level {level}");
        assert!(ours_decode(&out).is_empty());

        // flush before any write, then finish.
        let mut comp = compressor(level);
        comp.flush().unwrap();
        let out = comp.finish().unwrap().into_inner();
        assert!(c_decode_all(&out).is_empty());

        // A single byte.
        let mut comp = compressor(level);
        comp.write_all(b"x").unwrap();
        let out = comp.finish().unwrap().into_inner();
        assert_eq!(c_decode_all(&out), b"x");
        assert_eq!(ours_decode(&out), b"x");

        // Flush with nothing pending between writes must be harmless.
        let mut comp = compressor(level);
        comp.write_all(b"abc").unwrap();
        comp.flush().unwrap();
        comp.flush().unwrap();
        comp.write_all(b"def").unwrap();
        comp.flush().unwrap();
        comp.flush().unwrap();
        let out = comp.finish().unwrap().into_inner();
        assert_eq!(c_decode_all(&out), b"abcdef");
    }
}

#[test]
fn multi_megabyte_stream_exceeds_window() {
    // Level 1's window is 512 KiB, level 3's 1 MiB at the unknown-size
    // tier; 12 MiB written in odd-sized pieces forces many blocks, several
    // window slides and table rebases.
    for level in [1i32, 3, 9] {
        let data = text(12 << 20, 7);
        let mut comp = compressor(level);
        let mut pos = 0;
        let mut piece = 1usize;
        while pos < data.len() {
            let n = piece.min(data.len() - pos);
            comp.write_all(&data[pos..pos + n]).unwrap();
            pos += n;
            piece = (piece * 3 + 7) % 300_000 + 1;
            if pos % 5 == 0 {
                comp.flush().unwrap();
            }
        }
        let out = comp.finish().unwrap().into_inner();
        assert_eq!(c_decode_all(&out), data, "level {level}");
        assert_eq!(ours_decode(&out), data, "level {level}");
        // It should actually compress (text).
        assert!(out.len() < data.len() / 3, "level {level}: {} bytes for {}", out.len(), data.len());
    }
}

#[test]
fn incompressible_and_rle_blocks() {
    let mut random = vec![0u8; 400_000];
    let mut x = 0x9E3779B9u32;
    for b in random.iter_mut() {
        x ^= x << 13;
        x ^= x >> 17;
        x ^= x << 5;
        *b = (x >> 24) as u8;
    }
    let zeros = vec![0u8; 300_000];
    let mut comp = compressor(3);
    comp.write_all(&random).unwrap();
    comp.flush().unwrap();
    assert_eq!(c_decode_prefix(&emitted(&comp)), random);
    comp.write_all(&zeros).unwrap();
    comp.flush().unwrap();
    assert_eq!(c_decode_prefix(&emitted(&comp)), [random.clone(), zeros.clone()].concat());
    let out = comp.finish().unwrap().into_inner();
    assert_eq!(c_decode_all(&out), [random, zeros].concat());
}

#[test]
fn level_zero_is_raw_blocks() {
    let data = text(200_000, 5);
    let mut comp = compressor(0);
    comp.write_all(&data).unwrap();
    comp.flush().unwrap();
    assert_eq!(c_decode_prefix(&emitted(&comp)), data);
    let out = comp.finish().unwrap().into_inner();
    assert_eq!(c_decode_all(&out), data);
    assert!(out.len() >= data.len());
}

#[test]
fn stream_output_decodes_with_our_decoder_at_every_flush_point() {
    // Same as the C check but through our own frame decoder.
    let a = text(150_000, 11);
    let b = text(1_000, 12);
    let mut comp = compressor(3);
    comp.write_all(&a).unwrap();
    comp.flush().unwrap();
    assert_eq!(ours_decode_prefix(&emitted(&comp)), a);
    comp.write_all(&b).unwrap();
    let out = comp.finish().unwrap().into_inner();
    assert_eq!(ours_decode(&out), [a, b].concat());
}
