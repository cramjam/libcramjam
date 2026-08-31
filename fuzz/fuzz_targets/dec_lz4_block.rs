#![no_main]
mod common;
use libfuzzer_sys::fuzz_target;

fuzz_target!(|data: &[u8]| {
    // Prepended-size form (cramjam's decompress_block default).
    if data.len() >= 4 {
        let n = u32::from_le_bytes(data[..4].try_into().unwrap()) as usize;
        if n <= common::OUT_CAP {
            let r_ours = libcramjam::lz4::block::decompress_vec(data);
            let r_ref = lz4::block::decompress(&data[4..], Some(n as i32));
            common::check_differential("lz4_block", r_ours, r_ref);
        }
    }
    // Raw form into a fixed buffer (store_size=False path).
    let mut out = vec![0u8; 1 << 16];
    let mut refout = vec![0u8; 1 << 16];
    let r_ours = libcramjam::lz4::block::decompress_into(data, &mut out, Some(false));
    let r_ref = lz4::block::decompress_to_buffer(data, Some(refout.len() as i32), &mut refout);
    if let (Ok(a), Ok(b)) = (&r_ours, &r_ref) {
        assert!(out[..*a] == refout[..*b], "lz4_block raw: bytes differ");
    }
    if r_ours.is_err() && r_ref.is_ok() {
        panic!("lz4_block raw: C decoded {} bytes, ours errored", r_ref.unwrap());
    }
});
