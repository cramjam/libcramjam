//! Cross-implementation tests: our pure-Rust bzip2 decoder against C-backed `bzip2`.

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

/// Encode with the C bzip2 reference and decode with our native decoder.
fn c_bzip2_compress(data: &[u8], level: u32) -> Vec<u8> {
    let mut enc = bzip2::write::BzEncoder::new(Vec::new(), bzip2::Compression::new(level));
    enc.write_all(data).unwrap();
    enc.finish().unwrap()
}

fn c_bzip2_decompress(data: &[u8]) -> Vec<u8> {
    use std::io::Read;
    let mut dec = bzip2::read::BzDecoder::new(data);
    let mut out = Vec::new();
    dec.read_to_end(&mut out).unwrap();
    out
}

fn our_compress(data: &[u8], level: u32) -> Vec<u8> {
    libcramjam::bzip2_impl::encode::encode_stream(data, level)
}

fn our_decompress(data: &[u8]) -> Vec<u8> {
    let mut out = Vec::new();
    libcramjam::bzip2_impl::decode::decode_stream(data, &mut out).unwrap();
    out
}

#[test]
fn c_compress_our_decompress_tiny() {
    let data = b"abc".to_vec();
    let compressed = c_bzip2_compress(&data, 1);
    let decompressed = our_decompress(&compressed);
    assert_eq!(decompressed, data);
}

#[test]
fn c_compress_our_decompress_text() {
    for level in [1, 5, 9] {
        for size in [10, 100, 1_000, 10_000, 100_000, 500_000] {
            let data = gen_text(size);
            let compressed = c_bzip2_compress(&data, level);
            let decompressed = our_decompress(&compressed);
            assert_eq!(
                decompressed, data,
                "round-trip failed: level={}, size={}",
                level, size
            );
        }
    }
}

#[test]
fn c_compress_our_decompress_multi_block() {
    // Force >100 KB so a level=1 block size of 100 KB will produce multiple blocks.
    let data = gen_text(750_000);
    let compressed = c_bzip2_compress(&data, 1);
    let decompressed = our_decompress(&compressed);
    assert_eq!(decompressed, data);
}

#[test]
fn our_roundtrip_tiny() {
    let data = b"abc".to_vec();
    let compressed = our_compress(&data, 1);
    let decompressed = our_decompress(&compressed);
    assert_eq!(decompressed, data);
}

#[test]
fn our_compress_c_decompress_tiny() {
    let data = b"hello world".to_vec();
    let compressed = our_compress(&data, 1);
    let decompressed = c_bzip2_decompress(&compressed);
    assert_eq!(decompressed, data);
}

#[test]
fn our_compress_c_decompress_text() {
    for level in [1, 5, 9] {
        for size in [10, 100, 1_000, 10_000, 100_000] {
            let data = gen_text(size);
            let compressed = our_compress(&data, level);
            let decompressed = c_bzip2_decompress(&compressed);
            assert_eq!(
                decompressed, data,
                "ours→C round-trip failed: level={}, size={}",
                level, size
            );
        }
    }
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
    use std::str::FromStr;
    let bytes = read_dir_files(std::path::PathBuf::from_str("./src").unwrap());
    for level in [1, 5, 9] {
        let compressed = c_bzip2_compress(&bytes, level);
        let decompressed = our_decompress(&compressed);
        assert_eq!(decompressed, bytes, "src dir round-trip failed: level={}", level);
    }
}

#[test]
fn c_compress_our_decompress_repeated() {
    let data = vec![0xAAu8; 5000];
    let compressed = c_bzip2_compress(&data, 6);
    let decompressed = our_decompress(&compressed);
    assert_eq!(decompressed, data);
}

#[test]
fn c_compress_our_decompress_random() {
    let mut s: u32 = 0xCAFE_BABE;
    let data: Vec<u8> = (0..2000)
        .map(|_| {
            s ^= s << 13;
            s ^= s >> 17;
            s ^= s << 5;
            (s >> 16) as u8
        })
        .collect();
    let compressed = c_bzip2_compress(&data, 6);
    let decompressed = our_decompress(&compressed);
    assert_eq!(decompressed, data);
}
