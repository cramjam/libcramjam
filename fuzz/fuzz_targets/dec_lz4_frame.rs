#![no_main]
mod common;
use libfuzzer_sys::fuzz_target;

fuzz_target!(|data: &[u8]| {
    let mut ours = common::CappedVec::new();
    let r_ours = libcramjam::lz4::decompress(data, &mut ours).map(|_| ours.buf);
    let r_ref = lz4::Decoder::new(data).and_then(|mut d| {
        let out = common::read_capped(&mut d)?;
        d.finish().1?;
        Ok(out)
    });
    common::check_differential("lz4_frame", r_ours, r_ref);
});
