#![no_main]
//! Streaming compressors under arbitrary write/flush patterns: the finished
//! stream must decode (C reference) to exactly what was written.
mod common;
use libfuzzer_sys::fuzz_target;
use std::io::Write;

/// data[0] = codec, data[1] = level seed, then repeating [chunk_lo, chunk_hi,
/// op] groups: op & 1 = flush after write; chunk bytes come from the payload
/// tail.
fuzz_target!(|data: &[u8]| {
    if data.len() < 4 {
        return;
    }
    let codec = data[0] % 5;
    let level = data[1];
    let mut script = &data[2..];
    let mut fed: Vec<u8> = Vec::new();
    macro_rules! run {
        ($enc:expr, $dec:expr) => {{
            let mut enc = $enc;
            while script.len() >= 3 {
                let n = u16::from_le_bytes([script[0], script[1]]) as usize % 4096;
                let flush = script[2] & 1 == 1;
                script = &script[3..];
                let take = n.min(script.len());
                enc.write_all(&script[..take]).unwrap();
                fed.extend_from_slice(&script[..take]);
                script = &script[take..];
                if flush {
                    enc.flush().unwrap();
                }
            }
            let out: Vec<u8> = enc.finish().unwrap();
            let dec: Vec<u8> = $dec(&out[..]);
            assert!(dec == fed, "stream roundtrip mismatch (codec {})", codec);
        }};
    }
    match codec {
        0 => run!(
            libcramjam::zlib::ZlibStreamCompressor::new(Vec::new(), (level % 10) as u32),
            |o: &[u8]| common::read_capped(flate2::read::ZlibDecoder::new(o)).unwrap()
        ),
        1 => run!(
            libcramjam::zstd::ZstdStreamCompressor::new(Vec::new(), (level % 23) as i32).unwrap(),
            |o: &[u8]| zstd::stream::decode_all(o).unwrap()
        ),
        2 => run!(
            libcramjam::lz4::Lz4StreamCompressor::new(Vec::new(), (level % 13) as u32),
            |o: &[u8]| {
                let mut d = lz4::Decoder::new(o).unwrap();
                let r = common::read_capped(&mut d).unwrap();
                d.finish().1.unwrap();
                r
            }
        ),
        3 => run!(
            libcramjam::bzip2::Bzip2StreamCompressor::new(Vec::new(), (level % 9 + 1) as u32),
            |o: &[u8]| common::read_capped(bzip2::read::MultiBzDecoder::new(o)).unwrap()
        ),
        _ => run!(
            libcramjam::xz::XzStreamCompressor::new(Vec::new(), (level % 10) as u32),
            |o: &[u8]| common::read_capped(xz2::read::XzDecoder::new(o)).unwrap()
        ),
    }
});
