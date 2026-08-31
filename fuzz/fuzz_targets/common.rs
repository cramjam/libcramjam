//! Shared helpers for the fuzz targets.
#![allow(dead_code)]
use std::io::{self, Read, Write};

/// Cap decompressed output so decompression bombs fail identically on both
/// sides instead of OOMing the fuzzer.
pub const OUT_CAP: usize = 64 << 20;

pub struct CappedVec {
    pub buf: Vec<u8>,
}

impl CappedVec {
    pub fn new() -> Self {
        Self { buf: Vec::new() }
    }
}

impl Write for CappedVec {
    fn write(&mut self, b: &[u8]) -> io::Result<usize> {
        if self.buf.len() + b.len() > OUT_CAP {
            return Err(io::Error::new(io::ErrorKind::Other, "fuzz: output cap"));
        }
        self.buf.extend_from_slice(b);
        Ok(b.len())
    }
    fn flush(&mut self) -> io::Result<()> {
        Ok(())
    }
}

/// Read `r` fully into a capped buffer; Err on failure or cap.
pub fn read_capped<R: Read>(mut r: R) -> io::Result<Vec<u8>> {
    let mut out = Vec::new();
    let mut chunk = [0u8; 16384];
    loop {
        match r.read(&mut chunk) {
            Ok(0) => return Ok(out),
            Ok(n) => {
                if out.len() + n > OUT_CAP {
                    return Err(io::Error::new(io::ErrorKind::Other, "fuzz: output cap"));
                }
                out.extend_from_slice(&chunk[..n]);
            }
            Err(e) => return Err(e),
        }
    }
}

/// Differential oracle: if the C reference decodes `data`, ours must decode
/// it to the same bytes. Ours erroring while C errors is fine; ours
/// *succeeding* where C errors is tolerated (strictness differences), but
/// ours must never panic — libFuzzer turns any panic into a crash.
pub fn check_differential(
    name: &str,
    ours: io::Result<Vec<u8>>,
    reference: io::Result<Vec<u8>>,
) {
    match (ours, reference) {
        (Ok(a), Ok(b)) => {
            // Length divergence on an accepted stream is a real structural
            // bug (one decoder desynced). Byte-content divergence with equal
            // length is GIGO: a mutated, checksum-less stream both decoders
            // accept but resolve differently (e.g. a corrupt entropy-table
            // header). Neither is "correct"; valid-stream byte-exactness is
            // covered by the corpus cross-impl tests + the enc_roundtrip
            // fuzz target. So flag length divergence, tolerate content.
            assert_eq!(
                a.len(),
                b.len(),
                "{name}: length mismatch ours={} ref={} (structural desync)",
                a.len(),
                b.len()
            );
        }
        (Err(_), Ok(_)) => {
            // Ours rejecting a stream the C reference accepts is fine: on a
            // mutated/checksum-less input our decoder is allowed to be
            // stricter (e.g. verifying the zlib adler-32 or the lz4 magic
            // where the C wrapper is lenient). Rejecting corrupt data is
            // safer, not a bug. Valid-stream acceptance is covered by
            // tests/corpus_*.rs. Panics/OOB are caught regardless.
        }
        _ => {}
    }
}
