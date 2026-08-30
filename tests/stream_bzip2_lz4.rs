//! Streaming compressors must behave like the C crates' writers: `flush`
//! makes everything written so far decodable, `finish` ends the stream.
use std::io::{Cursor, Read, Write};

fn data(n: usize, seed: u32) -> Vec<u8> {
    let mut v = Vec::with_capacity(n);
    let mut x = seed.wrapping_mul(2654435761) | 1;
    let words = [b"alpha ", b"beta  ", b"gamma ", b"delta ", b"omega "];
    while v.len() < n {
        x ^= x << 13; x ^= x >> 17; x ^= x << 5;
        if x % 7 == 0 { v.push((x >> 24) as u8); } else { v.extend_from_slice(words[(x % 5) as usize]); }
    }
    v.truncate(n);
    v
}

/// Read as much as a streaming decoder can get from a possibly truncated
/// stream (ignore the trailing EOF/truncation error).
fn read_prefix<R: Read>(mut r: R) -> Vec<u8> {
    let mut out = Vec::new();
    let mut buf = [0u8; 8192];
    loop {
        match r.read(&mut buf) {
            Ok(0) => break,
            Ok(n) => out.extend_from_slice(&buf[..n]),
            Err(_) => break,
        }
    }
    out
}

#[test]
fn bzip2_stream_flush_and_finish() {
    for level in [1u32, 6, 9] {
        for &(na, nb, nc) in &[(0usize, 0usize, 0usize), (1, 0, 0), (1000, 1, 1), (250_000, 700_000, 50), (1_300_000, 10, 900_000)] {
            let (a, b, c) = (data(na, 1), data(nb, 2), data(nc, 3));
            let mut enc = libcramjam::bzip2::Bzip2StreamCompressor::new(Cursor::new(Vec::new()), level);
            enc.flush().unwrap(); // nothing pending: legal no-op
            enc.write_all(&a).unwrap();
            enc.flush().unwrap();
            let after_a = enc.get_ref().get_ref().clone();
            // libbzip2's BZ_FLUSH emits whole bytes only (the block's last
            // <8 bits stay queued — blocks are not byte-aligned), so a
            // streaming decoder sees a prefix; the block completes with the
            // next bytes. Same as the C-backed writer.
            let dec = bzip2::read::BzDecoder::new(&after_a[..]);
            let prefix = read_prefix(dec);
            assert!(a.starts_with(&prefix), "level {level} sizes {na}/{nb}/{nc}");
            assert!(na == 0 || after_a.len() > 4, "flush wrote the block: {} bytes for {na}", after_a.len());
            enc.write_all(&b).unwrap();
            enc.flush().unwrap();
            enc.write_all(&c).unwrap();
            let out = enc.finish().unwrap().into_inner();
            let mut all = Vec::new();
            bzip2::read::BzDecoder::new(&out[..]).read_to_end(&mut all).unwrap();
            assert_eq!(all, [a.clone(), b.clone(), c.clone()].concat());
            let mut ours = Vec::new();
            libcramjam::bzip2::decompress(&mut Cursor::new(&out), &mut ours).unwrap();
            assert_eq!(ours, all);
        }
    }
}

#[test]
fn lz4_stream_flush_and_finish() {
    for level in [1u32, 2, 4, 9] {
        for &(linked, checksum) in &[(true, true), (true, false), (false, true)] {
            for &(na, nb, nc) in &[(0usize, 0usize, 0usize), (1, 0, 0), (1000, 1, 1), (70_000, 200_000, 50), (2_500_000, 10, 300_000)] {
                let (a, b, c) = (data(na, 4), data(nb, 5), data(nc, 6));
                let mut enc = libcramjam::lz4::Lz4StreamCompressor::with_options(Cursor::new(Vec::new()), level, linked, checksum);
                enc.flush().unwrap();
                enc.write_all(&a).unwrap();
                enc.flush().unwrap();
                let after_a = enc.get_ref().get_ref().clone();
                let dec = lz4::Decoder::new(&after_a[..]).unwrap();
                assert_eq!(read_prefix(dec), a, "level {level} linked {linked} sizes {na}/{nb}/{nc}");
                enc.write_all(&b).unwrap();
                enc.flush().unwrap();
                enc.write_all(&c).unwrap();
                let out = enc.finish().unwrap().into_inner();
                let mut dec = lz4::Decoder::new(&out[..]).unwrap();
                let mut all = Vec::new();
                dec.read_to_end(&mut all).unwrap();
                let (_, res) = dec.finish();
                res.unwrap(); // C verifies the content checksum here
                assert_eq!(all, [a.clone(), b.clone(), c.clone()].concat());
                let mut ours = Vec::new();
                libcramjam::lz4::decompress(&mut Cursor::new(&out), &mut ours).unwrap();
                assert_eq!(ours, all);
            }
        }
    }
}
