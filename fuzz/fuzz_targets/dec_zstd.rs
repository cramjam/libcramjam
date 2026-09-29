#![no_main]
mod common;
use libfuzzer_sys::fuzz_target;

fuzz_target!(|data: &[u8]| {
    let mut ours = common::CappedVec::new();
    let r_ours = libcramjam::zstd::decompress(data, &mut ours).map(|_| ours.buf);
    let r_ref = zstd::stream::read::Decoder::new(data)
        .and_then(common::read_capped);
    common::check_differential("zstd", r_ours, r_ref);
});
