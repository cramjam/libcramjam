use std::io::{Cursor, Read, Write};

#[path = "../benches/common.rs"]
mod common;

fn ours(d: &[u8], l: u32) -> Vec<u8> {
    let mut o = Vec::new();
    libcramjam::deflate::compress(&mut Cursor::new(d), &mut o, Some(l)).unwrap();
    o
}
fn f2_dec(d: &[u8]) -> Result<Vec<u8>, std::io::Error> {
    let mut dec = flate2::read::DeflateDecoder::new(d);
    let mut o = Vec::new();
    dec.read_to_end(&mut o)?;
    Ok(o)
}
fn ours_dec(d: &[u8]) -> Vec<u8> {
    let mut o = Vec::new();
    libcramjam::deflate::decompress(&mut Cursor::new(d), &mut o).unwrap();
    o
}

fn main() {
    for (name, data) in common::load_all() {
        for level in [1u32, 6, 9] {
            let o = ours(data, level);
            let f2 = f2_dec(&o);
            let ours_rt = ours_dec(&o);
            let ok_self = ours_rt == data;
            match f2 {
                Ok(d) if d == data => {},
                Ok(d) => println!("{} L{}: OURS->F2 BYTES MISMATCH (got {} bytes, want {})", name, level, d.len(), data.len()),
                Err(e) => println!("{} L{}: OURS->F2 ERR: {} (self_rt={})", name, level, e, ok_self),
            }
        }
    }
    println!("done");
}
