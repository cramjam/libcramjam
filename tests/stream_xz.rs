//! Streaming `XzStreamCompressor` semantics (liblzma `LZMA_RUN` /
//! `LZMA_SYNC_FLUSH` / `LZMA_FINISH`), verified through liblzma's decoder.

use std::io::{Read, Write};

use libcramjam::xz::XzStreamCompressor;

fn liblzma_decode(data: &[u8]) -> Vec<u8> {
    let mut out = Vec::new();
    xz2::read::XzDecoder::new(data)
        .read_to_end(&mut out)
        .unwrap();
    out
}

/// Decode as far as the stream currently goes (after a sync flush the
/// stream is truncated: no end marker / index yet). Returns what a
/// streaming decoder can hand back from the prefix.
fn liblzma_decode_prefix(data: &[u8]) -> Vec<u8> {
    let mut out = Vec::new();
    let mut dec = xz2::read::XzDecoder::new(data);
    let mut buf = [0u8; 4096];
    loop {
        match dec.read(&mut buf) {
            Ok(0) => break,
            Ok(n) => out.extend_from_slice(&buf[..n]),
            Err(e) if e.kind() == std::io::ErrorKind::UnexpectedEof => break,
            Err(e) => panic!("decode error: {e}"),
        }
    }
    out
}

fn text(n: usize, seed: u64) -> Vec<u8> {
    let words: &[&str] = &[
        "the ", "quick ", "brown ", "fox ", "jumps ", "over ", "lazy ", "dog ", "lorem ", "ipsum ",
        "\n",
    ];
    let mut s = seed | 1;
    let mut v = Vec::with_capacity(n);
    while v.len() < n {
        s ^= s << 13;
        s ^= s >> 7;
        s ^= s << 17;
        v.extend_from_slice(words[(s % words.len() as u64) as usize].as_bytes());
    }
    v.truncate(n);
    v
}

fn random(n: usize, seed: u64) -> Vec<u8> {
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

fn one_shot(input: &[u8], preset: u32) -> Vec<u8> {
    let mut out = Vec::new();
    libcramjam::xz::compress(
        &mut std::io::Cursor::new(input),
        &mut out,
        Some(preset),
        None::<libcramjam::xz::Format>,
        None::<libcramjam::xz::Check>,
        None::<libcramjam::xz::Filters>,
        None::<libcramjam::xz::LzmaOptions>,
    )
    .unwrap();
    out
}

#[test]
fn write_flush_write_flush_write_finish_decodes_via_liblzma() {
    for preset in [0u32, 1, 6, 9] {
        let a = text(70_000, 1);
        let b = random(3_000, 2);
        let c = text(200_000, 3);
        let mut enc = XzStreamCompressor::new(Vec::new(), preset);
        enc.write_all(&a).unwrap();
        enc.flush().unwrap();
        let after_a = enc.get_ref().len();
        // Prefix after the first flush decodes exactly A.
        assert_eq!(
            liblzma_decode_prefix(enc.get_ref()),
            a,
            "preset {preset}: prefix after flush"
        );
        enc.write_all(&b).unwrap();
        enc.flush().unwrap();
        assert!(enc.get_ref().len() > after_a);
        let mut ab = a.clone();
        ab.extend_from_slice(&b);
        assert_eq!(
            liblzma_decode_prefix(enc.get_ref()),
            ab,
            "preset {preset}: prefix after 2nd flush"
        );
        enc.write_all(&c).unwrap();
        let out = enc.finish().unwrap();
        let mut abc = ab;
        abc.extend_from_slice(&c);
        assert_eq!(liblzma_decode(&out), abc, "preset {preset}: full stream");
        // Our decoder agrees.
        let mut ours = Vec::new();
        libcramjam::xz::decompress(&mut std::io::Cursor::new(&out), &mut ours).unwrap();
        assert_eq!(ours, abc);
    }
}

#[test]
fn flush_with_nothing_pending_is_a_no_op() {
    let a = text(10_000, 7);
    let mut enc = XzStreamCompressor::new(Vec::new(), 6);
    enc.flush().unwrap();
    enc.flush().unwrap();
    enc.write_all(&a).unwrap();
    enc.flush().unwrap();
    let n = enc.get_ref().len();
    enc.flush().unwrap();
    assert_eq!(enc.get_ref().len(), n);
    let out = enc.finish().unwrap();
    assert_eq!(liblzma_decode(&out), a);
}

#[test]
fn empty_and_tiny_inputs() {
    for preset in [0u32, 6] {
        let out = XzStreamCompressor::new(Vec::new(), preset)
            .finish()
            .unwrap();
        assert_eq!(liblzma_decode(&out), b"");
        assert_eq!(out, one_shot(b"", preset));

        let mut enc = XzStreamCompressor::new(Vec::new(), preset);
        enc.write_all(b"x").unwrap();
        let out = enc.finish().unwrap();
        assert_eq!(liblzma_decode(&out), b"x");
        assert_eq!(out, one_shot(b"x", preset));

        // One byte at a time, flush after each.
        let mut enc = XzStreamCompressor::new(Vec::new(), preset);
        for b in b"hello" {
            enc.write_all(&[*b]).unwrap();
            enc.flush().unwrap();
        }
        let out = enc.finish().unwrap();
        assert_eq!(liblzma_decode(&out), b"hello");
    }
}

#[test]
fn single_write_matches_one_shot_bytes() {
    for preset in [0u32, 1, 6, 9] {
        let a = text(150_000, 11);
        let mut enc = XzStreamCompressor::new(Vec::new(), preset);
        enc.write_all(&a).unwrap();
        let out = enc.finish().unwrap();
        assert_eq!(out, one_shot(&a, preset), "preset {preset}");
    }
}

#[test]
fn many_small_writes_decode() {
    let a = text(1_000_000, 5);
    let mut enc = XzStreamCompressor::new(Vec::new(), 6);
    for chunk in a.chunks(777) {
        enc.write_all(chunk).unwrap();
    }
    let out = enc.finish().unwrap();
    assert_eq!(liblzma_decode(&out), a);
    // Sliding-window streaming should compress about as well as one-shot.
    let one = one_shot(&a, 6).len();
    assert!(
        out.len() as f64 <= one as f64 * 1.02,
        "stream {} vs one-shot {}",
        out.len(),
        one
    );
}

#[test]
fn multi_mb_exceeding_dict_size_slides_window() {
    // preset 1 dict = 1 MiB, preset 6 dict = 8 MiB; feed more than both,
    // with periodic flushes, and data that repeats at a distance larger
    // than the dictionary so far matches are actually discarded.
    let block = text(3 << 20, 9);
    let mut input = Vec::new();
    for i in 0..4u8 {
        input.extend_from_slice(&block);
        input.extend_from_slice(&random(64 << 10, 100 + i as u64));
    }
    for preset in [1u32, 6] {
        let mut enc = XzStreamCompressor::new(Vec::new(), preset);
        for (i, chunk) in input.chunks(1 << 20).enumerate() {
            enc.write_all(chunk).unwrap();
            if i % 3 == 2 {
                enc.flush().unwrap();
            }
        }
        let out = enc.finish().unwrap();
        assert_eq!(liblzma_decode(&out), input, "preset {preset}");
        let mut ours = Vec::new();
        libcramjam::xz::decompress(&mut std::io::Cursor::new(&out), &mut ours).unwrap();
        assert_eq!(ours, input);
    }
}

#[test]
fn incompressible_data_uses_stored_chunks() {
    let a = random(300_000, 42);
    let mut enc = XzStreamCompressor::new(Vec::new(), 6);
    enc.write_all(&a[..100_000]).unwrap();
    enc.flush().unwrap();
    enc.write_all(&a[100_000..]).unwrap();
    let out = enc.finish().unwrap();
    assert_eq!(liblzma_decode(&out), a);
    assert!(out.len() < a.len() + a.len() / 100 + 200);
}

#[test]
fn invalid_preset_errors_on_use() {
    let mut enc = XzStreamCompressor::new(Vec::new(), 42);
    assert!(enc.write_all(b"abc").is_err());
}

/// Regression (fuzzer-found): at presets 4-9 (bt4 + optimum parser) a sync
/// flush landing at certain data positions used to leave stale match-finder
/// tree state that produced a false cross-chunk match in the next chunk,
/// silently corrupting the stream (both our decoder and liblzma rejected the
/// output). Fixed by resetting the match finder at each sync flush. The
/// fixture + op pattern is the minimized fuzzer case (first bad byte 4500).
#[test]
fn streaming_flush_bt4_no_false_match() {
    let fed = include_bytes!("fixtures/xz_stream_flush_bt4.bin");
    let ops: &[(usize, bool)] = &[
        (353, true), (353, true), (353, true), (353, true), (353, true),
        (353, true), (353, true), (353, true), (32, false), (58, true),
        (353, true), (353, true), (353, true), (32, false), (1838, true),
    ];
    for preset in [1u32, 3, 4, 6, 8, 9] {
        let mut enc = XzStreamCompressor::new(Vec::new(), preset);
        let mut pos = 0usize;
        for &(n, flush) in ops {
            let e = (pos + n).min(fed.len());
            enc.write_all(&fed[pos..e]).unwrap();
            pos = e;
            if flush {
                enc.flush().unwrap();
            }
        }
        if pos < fed.len() {
            enc.write_all(&fed[pos..]).unwrap();
        }
        let out = enc.finish().unwrap();
        let mut dec = Vec::new();
        xz2::read::XzDecoder::new(&out[..])
            .read_to_end(&mut dec)
            .unwrap_or_else(|e| panic!("preset {preset}: liblzma rejected our stream: {e}"));
        assert_eq!(dec.len(), fed.len(), "preset {preset} length");
        assert!(dec == fed, "preset {preset}: streaming xz output corrupted");
    }
}
