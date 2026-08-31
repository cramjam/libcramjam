#![no_main]
mod common;
use libfuzzer_sys::fuzz_target;

fuzz_target!(|data: &[u8]| {
    // .xz container (multi-stream, like liblzma's auto decoder).
    let mut ours = common::CappedVec::new();
    let r_ours = libcramjam::xz::decompress(data, &mut ours).map(|_| ours.buf);

    // The C reference (liblzma) honours the stream's declared dictionary
    // size, so a crafted 4 GiB dict_size makes it malloc 4 GiB and OOM the
    // fuzzer — that is liblzma's own memory use, not a libcramjam bug (our
    // decoder uses the output as its dictionary and never allocates it).
    // Bound liblzma with a memlimit so it errors instead; a fresh stream
    // per call keeps the multi-stream decode behaviour.
    let r_ref = xz2::stream::Stream::new_stream_decoder(512 << 20, 0)
        .map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e))
        .and_then(|s| common::read_capped(xz2::read::XzDecoder::new_stream(data, s)));

    common::check_differential("xz", r_ours, r_ref);
});
