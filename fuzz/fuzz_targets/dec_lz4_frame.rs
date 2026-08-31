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
    // `libcramjam::lz4::decompress` decodes *concatenated* frames (lz4 CLI
    // behaviour); the `lz4` crate Decoder stops after one, so on multi-frame
    // input ours is legitimately longer. Require only that ours begins with
    // C's bytes (exact prefix) — that catches real corruption, not the
    // extra frames.
    match (r_ours, r_ref) {
        (Ok(a), Ok(b)) => assert!(a.len() >= b.len() && a[..b.len()] == b[..], "lz4_frame: prefix diverges from C"),
        _ => {}
    }
});
