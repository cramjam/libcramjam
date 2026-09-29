#![no_main]
//! Encoders at arbitrary levels: our compress must round-trip through BOTH
//! our decoder and the C reference decoder.
mod common;
use libfuzzer_sys::fuzz_target;
use std::io::Cursor;

fn ours_and_ref(codec: u8, level: u8, payload: &[u8]) {
    let mut comp = Vec::new();
    match codec {
        0 => {
            libcramjam::zlib::compress(payload, &mut comp, Some(level as u32 % 10)).unwrap();
            let ours = common::read_capped(flate2::read::ZlibDecoder::new(&comp[..])).unwrap();
            assert!(ours == payload, "zlib: C decoder mismatch");
        }
        1 => {
            libcramjam::gzip::compress(payload, &mut comp, Some(level as u32 % 10)).unwrap();
            let ours = common::read_capped(flate2::read::MultiGzDecoder::new(&comp[..])).unwrap();
            assert!(ours == payload, "gzip: C decoder mismatch");
        }
        2 => {
            libcramjam::zstd::compress(payload, &mut comp, Some(level as i32 % 23), Some(payload.len())).unwrap();
            let r = zstd::stream::decode_all(&comp[..]).unwrap();
            assert!(r == payload, "zstd: C decoder mismatch");
        }
        3 => {
            libcramjam::lz4::compress(payload, &mut comp, Some(level as u32 % 13)).unwrap();
            let mut d = lz4::Decoder::new(&comp[..]).unwrap();
            let r = common::read_capped(&mut d).unwrap();
            d.finish().1.unwrap();
            assert!(r == payload, "lz4: C decoder mismatch");
        }
        4 => {
            libcramjam::bzip2::compress(payload, &mut comp, Some(level as u32 % 9 + 1)).unwrap();
            let r = common::read_capped(bzip2::read::MultiBzDecoder::new(&comp[..])).unwrap();
            assert!(r == payload, "bzip2: C decoder mismatch");
        }
        _ => {
            libcramjam::xz::compress(
                payload,
                &mut comp,
                Some(level as u32 % 10),
                None::<libcramjam::xz::Format>,
                None::<libcramjam::xz::Check>,
                None::<libcramjam::xz::Filters>,
                None::<libcramjam::xz::LzmaOptions>,
            )
            .unwrap();
            let r = common::read_capped(xz2::read::XzDecoder::new(&comp[..])).unwrap();
            assert!(r == payload, "xz: C decoder mismatch");
        }
    }
    // And through our own decoder.
    let mut back = common::CappedVec::new();
    match codec {
        0 => libcramjam::zlib::decompress(&comp[..], &mut back).unwrap(),
        1 => libcramjam::gzip::decompress(&comp[..], &mut back).unwrap(),
        2 => libcramjam::zstd::decompress(&comp[..], &mut back).unwrap(),
        3 => libcramjam::lz4::decompress(&comp[..], &mut back).unwrap(),
        4 => libcramjam::bzip2::decompress(&comp[..], &mut back).unwrap(),
        _ => libcramjam::xz::decompress(&comp[..], &mut back).unwrap(),
    };
    assert!(back.buf == payload, "our decoder mismatch");
    // lz4 block forms round-trip too.
    if codec == 3 {
        let comp = libcramjam::lz4::block::compress_vec(payload, Some(level as u32 % 13), None, Some(true)).unwrap();
        let back = libcramjam::lz4::block::decompress_vec(&comp).unwrap();
        assert!(back == payload, "lz4 block roundtrip");
        let refd = lz4::block::decompress(&comp[4..], Some(payload.len() as i32)).unwrap();
        assert!(refd == payload, "lz4 block C decoder mismatch");
    }
    let _ = Cursor::new(comp);
}

fuzz_target!(|data: &[u8]| {
    if data.len() < 2 {
        return;
    }
    ours_and_ref(data[0] % 6, data[1], &data[2..]);
});
