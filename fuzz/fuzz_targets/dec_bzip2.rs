#![no_main]
mod common;
use libfuzzer_sys::fuzz_target;

fuzz_target!(|data: &[u8]| {
    let mut ours = common::CappedVec::new();
    let r_ours = libcramjam::bzip2::decompress(data, &mut ours).map(|_| ours.buf);
    let r_ref = common::read_capped(bzip2::read::MultiBzDecoder::new(data));
    common::check_differential("bzip2", r_ours, r_ref);
});
