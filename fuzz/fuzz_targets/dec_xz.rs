#![no_main]
mod common;
use libfuzzer_sys::fuzz_target;

fuzz_target!(|data: &[u8]| {
    // .xz container (multi-stream, like liblzma's auto decoder).
    let mut ours = common::CappedVec::new();
    let r_ours = libcramjam::xz::decompress(data, &mut ours).map(|_| ours.buf);
    let r_ref = common::read_capped(xz2::read::XzDecoder::new_multi_decoder(data));
    common::check_differential("xz", r_ours, r_ref);
});
