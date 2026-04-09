//! Cross-implementation tests for the pure-Rust XZ / LZMA codec.
//!
//! These compress with C `xz2` (still kept as a dev-dependency) and
//! decompress with our pure-Rust `xz_impl`.  Once the encoder lands we'll
//! also add ours→C round-trips here, mirroring the bzip2 test layout.

use std::io::Write;

fn gen_text(size: usize) -> Vec<u8> {
    let phrases: &[&[u8]] = &[
        b"The quick brown fox jumps over the lazy dog. ",
        b"Lorem ipsum dolor sit amet, consectetur adipiscing elit. ",
        b"fn main() { println!(\"hello world\"); }\n",
    ];
    let mut data = Vec::with_capacity(size);
    for phrase in phrases.iter().cycle() {
        let take = phrase.len().min(size - data.len());
        data.extend_from_slice(&phrase[..take]);
        if data.len() >= size {
            break;
        }
    }
    data
}

fn c_xz_compress(data: &[u8], level: u32) -> Vec<u8> {
    let mut enc = xz2::write::XzEncoder::new(Vec::new(), level);
    enc.write_all(data).unwrap();
    enc.finish().unwrap()
}

fn our_decompress(data: &[u8]) -> Vec<u8> {
    libcramjam::xz_impl::decode_xz(data).unwrap()
}

#[test]
fn c_compress_our_decompress_tiny() {
    let data = b"abc".to_vec();
    let compressed = c_xz_compress(&data, 6);
    let decompressed = our_decompress(&compressed);
    assert_eq!(decompressed, data);
}

#[test]
fn c_compress_our_decompress_hello() {
    let data = b"hello world".to_vec();
    let compressed = c_xz_compress(&data, 6);
    let decompressed = our_decompress(&compressed);
    assert_eq!(decompressed, data);
}

#[test]
fn c_compress_our_decompress_text_small() {
    for level in [1u32, 6, 9] {
        let data = gen_text(1_000);
        let compressed = c_xz_compress(&data, level);
        let decompressed = our_decompress(&compressed);
        assert_eq!(decompressed, data, "level={}", level);
    }
}

#[test]
fn c_compress_our_decompress_text_medium() {
    for level in [1u32, 6, 9] {
        let data = gen_text(100_000);
        let compressed = c_xz_compress(&data, level);
        let decompressed = our_decompress(&compressed);
        assert_eq!(decompressed, data, "level={}", level);
    }
}

#[test]
fn c_compress_our_decompress_repeated() {
    let data = vec![0xAAu8; 5000];
    let compressed = c_xz_compress(&data, 6);
    let decompressed = our_decompress(&compressed);
    assert_eq!(decompressed, data);
}

#[test]
fn c_compress_our_decompress_random_small() {
    let mut s: u32 = 0xCAFE_BABE;
    let data: Vec<u8> = (0..2000)
        .map(|_| {
            s ^= s << 13;
            s ^= s >> 17;
            s ^= s << 5;
            (s >> 16) as u8
        })
        .collect();
    let compressed = c_xz_compress(&data, 6);
    let decompressed = our_decompress(&compressed);
    assert_eq!(decompressed, data);
}

#[test]
fn c_compress_our_decompress_src_dir() {
    fn read_dir_files(dir: std::path::PathBuf) -> Vec<u8> {
        let mut all = Vec::new();
        for entry in std::fs::read_dir(dir).unwrap() {
            let entry = entry.unwrap();
            if entry.file_type().unwrap().is_file() {
                all.extend(std::fs::read(entry.path()).unwrap());
            } else if entry.file_type().unwrap().is_dir() {
                all.extend(read_dir_files(entry.path()));
            }
        }
        all
    }
    let bytes = read_dir_files(std::path::PathBuf::from("./src"));
    for level in [1u32, 5, 9] {
        let compressed = c_xz_compress(&bytes, level);
        let decompressed = our_decompress(&compressed);
        assert_eq!(decompressed, bytes, "src dir failed: level={}", level);
    }
}

#[test]
fn c_compress_our_decompress_text_large() {
    // 1 MiB of phrase-cycled text — triggers multi-block at preset 1.
    let data = gen_text(1_000_000);
    for level in [1u32, 6, 9] {
        let compressed = c_xz_compress(&data, level);
        let decompressed = our_decompress(&compressed);
        assert_eq!(decompressed, data, "level={}", level);
    }
}

#[test]
fn c_compress_our_decompress_random_large() {
    let mut s: u32 = 0xDEAD_BEEF;
    let data: Vec<u8> = (0..200_000)
        .map(|_| {
            s ^= s << 13;
            s ^= s >> 17;
            s ^= s << 5;
            (s >> 16) as u8
        })
        .collect();
    let compressed = c_xz_compress(&data, 6);
    let decompressed = our_decompress(&compressed);
    assert_eq!(decompressed, data);
}

#[test]
fn c_compress_our_decompress_empty() {
    let data: Vec<u8> = vec![];
    let compressed = c_xz_compress(&data, 6);
    let decompressed = our_decompress(&compressed);
    assert_eq!(decompressed, data);
}

#[test]
fn c_compress_our_decompress_single_byte() {
    let data = vec![0x42u8];
    let compressed = c_xz_compress(&data, 6);
    let decompressed = our_decompress(&compressed);
    assert_eq!(decompressed, data);
}
