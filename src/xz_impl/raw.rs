//! Raw filter-chain coding (liblzma's `lzma_raw_encoder` / `lzma_raw_decoder`,
//! Python's `lzma.FORMAT_RAW`): no container, no check — just the output of
//! the filter chain.  `[.., LZMA2]` is the LZMA2 chunk stream an .xz block
//! carries; `[.., LZMA1]` is one LZMA1 range-coded stream ending in the
//! end-of-payload marker.  BCJ filters run first when encoding and last
//! when decoding.

use std::io;

use super::options::{bcj_encode, split_chain, ResolvedFilter};

pub(crate) fn encode_raw(input: &[u8], chain: &[ResolvedFilter], output: &mut Vec<u8>) -> io::Result<()> {
    let (bcj, last) = split_chain(chain)?;
    let mut filtered = Vec::new();
    let data = bcj_encode(input, &bcj, &mut filtered);
    match last {
        ResolvedFilter::Lzma2(opts) => super::lzma_enc::encode_lzma_to_lzma2(data, opts, output),
        ResolvedFilter::Lzma1(opts) => super::lzma_enc::encode_lzma1_raw(data, opts, output),
        ResolvedFilter::Bcj(_) => unreachable!("split_chain guarantees an LZMA tail"),
    }
}

pub(crate) fn decode_raw(input: &[u8], chain: &[ResolvedFilter], output: &mut Vec<u8>) -> io::Result<()> {
    let (bcj, last) = split_chain(chain)?;
    let start = output.len();
    match last {
        ResolvedFilter::Lzma2(opts) => {
            super::lzma2::decode_lzma2(input, opts.dict_size.max(4096), output)?;
        }
        ResolvedFilter::Lzma1(opts) => {
            super::alone::decode_lzma1_stream(input, opts.lc, opts.lp, opts.pb, opts.dict_size.max(4096), None, output)?;
        }
        ResolvedFilter::Bcj(_) => unreachable!("split_chain guarantees an LZMA tail"),
    }
    for &id in bcj.iter().rev() {
        super::bcj::apply(id, &mut output[start..], 0, false);
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::xz_impl::options::{Filters, LzmaOptions};

    fn sample() -> Vec<u8> {
        let mut v: Vec<u8> = b"raw lzma round trip through every chain shape ".repeat(300);
        for i in 0..2000u32 {
            v.extend_from_slice(&[0xE8, (i & 0xFF) as u8, (i >> 8) as u8, 0, 0]);
        }
        v
    }

    #[test]
    fn raw_round_trips_all_chain_shapes() {
        let data = sample();
        let opts = LzmaOptions::new_preset(3).unwrap();
        let chains: Vec<Filters> = vec![
            Filters::new(),
            { let mut f = Filters::new(); f.lzma1(&opts); f },
            { let mut f = Filters::new(); f.x86().lzma2(&opts); f },
            { let mut f = Filters::new(); f.arm().lzma1(&opts); f },
            { let mut f = Filters::new(); f.sparc().powerpc().ia64().lzma2(&opts); f },
        ];
        for f in &chains {
            let chain = f.resolve(6, None).unwrap();
            let mut enc = Vec::new();
            encode_raw(&data, &chain, &mut enc).unwrap();
            let mut dec = Vec::new();
            decode_raw(&enc, &chain, &mut dec).unwrap();
            assert_eq!(dec, data, "{chain:?}");
        }
        // Empty input.
        let chain = Filters::new().resolve(6, None).unwrap();
        let mut enc = Vec::new();
        encode_raw(&[], &chain, &mut enc).unwrap();
        let mut dec = Vec::new();
        decode_raw(&enc, &chain, &mut dec).unwrap();
        assert!(dec.is_empty());
    }

    #[test]
    fn invalid_chains_are_rejected() {
        let opts = LzmaOptions::default();
        let mut f = Filters::new();
        f.lzma2(&opts).x86();
        assert!(f.resolve(6, None).is_err(), "LZMA before BCJ");
        let mut f = Filters::new();
        f.x86();
        assert!(f.resolve(6, None).is_err(), "no LZMA tail");
        let mut f = Filters::new();
        f.x86().arm().sparc().powerpc().lzma2(&opts);
        assert!(f.resolve(6, None).is_err(), "5 filters");
    }
}
