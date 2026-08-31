#![no_main]
mod common;
use libfuzzer_sys::fuzz_target;
use std::io::Cursor;

fuzz_target!(|data: &[u8]| {
    // zlib
    let mut ours = common::CappedVec::new();
    let r_ours = libcramjam::zlib::decompress(data, &mut ours).map(|_| ours.buf);
    let r_ref = common::read_capped(flate2::read::ZlibDecoder::new(data));
    common::check_differential("zlib", r_ours, r_ref);

    // gzip (multi-member, like cramjam)
    let mut ours = common::CappedVec::new();
    let r_ours = libcramjam::gzip::decompress(data, &mut ours).map(|_| ours.buf);
    let r_ref = common::read_capped(flate2::read::MultiGzDecoder::new(data));
    common::check_differential("gzip", r_ours, r_ref);

    // raw deflate
    let mut ours = common::CappedVec::new();
    let r_ours = libcramjam::deflate::decompress(Cursor::new(data), &mut ours).map(|_| ours.buf);
    let r_ref = common::read_capped(flate2::read::DeflateDecoder::new(data));
    common::check_differential("deflate", r_ours, r_ref);
});
