#![no_main]
//! Panic sweep for the *_into paths cramjam exposes: arbitrary data into an
//! arbitrary-size caller buffer must error cleanly, never panic.
mod common;
use libfuzzer_sys::fuzz_target;
use std::io::Cursor;

fuzz_target!(|data: &[u8]| {
    if data.len() < 3 {
        return;
    }
    let out_len = u16::from_le_bytes(data[..2].try_into().unwrap()) as usize;
    let sel = data[2] % 7;
    let payload = &data[3..];
    let mut out = vec![0u8; out_len];
    let mut cur = Cursor::new(&mut out[..]);
    let _ = match sel {
        0 => libcramjam::zlib::decompress(payload, &mut cur),
        1 => libcramjam::gzip::decompress(payload, &mut cur),
        2 => libcramjam::deflate::decompress(Cursor::new(payload), &mut cur),
        3 => libcramjam::zstd::decompress(payload, &mut cur),
        4 => libcramjam::lz4::decompress(payload, &mut cur),
        5 => libcramjam::bzip2::decompress(payload, &mut cur),
        _ => libcramjam::xz::decompress(payload, &mut cur),
    };
    let _ = libcramjam::lz4::block::decompress_into(payload, &mut out, Some(data[2] & 0x80 != 0));
});
