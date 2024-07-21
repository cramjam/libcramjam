#[cfg(feature = "blosc2")]
pub mod blosc2;
#[cfg(feature = "brotli")]
pub mod brotli;
#[cfg(feature = "bzip2")]
pub mod bzip2;
#[cfg(feature = "capi")]
mod capi;
#[cfg(feature = "deflate")]
pub mod deflate;
#[cfg(feature = "gzip")]
pub mod gzip;
#[cfg(feature = "lz4")]
pub mod lz4;
#[cfg(feature = "snappy")]
pub mod snappy;
#[cfg(feature = "xz")]
pub mod xz;
#[cfg(feature = "zstd")]
pub mod zstd;

#[cfg(test)]
mod tests {

    use std::io::Cursor;

    // Default testing data
    fn gen_data() -> Vec<u8> {
        (0..1_000_000)
            .map(|_| b"oh what a beautiful morning, oh what a beautiful day!!".to_vec())
            .flat_map(|v| v)
            .collect()
    }

    // Single test generation
    macro_rules! round_trip {
        ($name:ident($compress_output:ident -> $decompress_output:ident), variant=$variant:ident, header=$header:literal, $(, $args:ident)*) => {
            #[test]
            fn $name() {
                let data = gen_data();

                let mut compressed = Vec::new();

                let compressed_size = if stringify!($decompress_output) == "Slice" {
                        compressed = (0..data.len()).map(|_| 0).collect::<Vec<u8>>();
                        let mut cursor = Cursor::new(compressed.as_mut_slice());
                        crate::$variant::compress(&mut Cursor::new(data.as_slice()), &mut cursor $(, $args)*).unwrap()
                    } else {
                        crate::$variant::compress(&mut Cursor::new(data.as_slice()), &mut Cursor::new(&mut compressed) $(, $args)*).unwrap()
                    };

                compressed.truncate(compressed_size);
                assert_eq!(&compressed[..$header.len()], $header);

                let mut decompressed = Vec::new();

                let decompressed_size = if stringify!($decompress_output) == "Slice" {
                        decompressed = (0..data.len()).map(|_| 0).collect::<Vec<u8>>();
                        let mut cursor = Cursor::new(decompressed.as_mut_slice());
                        crate::$variant::decompress(&mut Cursor::new(&compressed), &mut cursor).unwrap()
                    } else {
                        crate::$variant::decompress(&mut Cursor::new(&compressed), &mut decompressed).unwrap()
                    };
                assert_eq!(decompressed_size, data.len());
                if &decompressed[..decompressed_size] != &data {
                    panic!("Decompressed and original data do not match! :-(")
                }
            }
        }
    }

    // macro to generate each variation of Output::* roundtrip.
    macro_rules! test_variant {
        ($variant:ident, header=$header:literal $(, $args:tt)*) => {
         #[cfg(test)]
         mod $variant {
            use super::*;
            round_trip!(roundtrip_compress_via_slice_decompress_via_slice(Slice -> Slice), variant=$variant, header=$header, $(, $args)* );
            round_trip!(roundtrip_compress_via_slice_decompress_via_vector(Slice -> Vector), variant=$variant, header=$header, $(, $args)* );
            round_trip!(roundtrip_compress_via_vector_decompress_via_slice(Vector -> Slice), variant=$variant, header=$header, $(, $args)* );
            round_trip!(roundtrip_compress_via_vector_decompress_via_vector(Vector -> Vector), variant=$variant, header=$header, $(, $args)* );
         }
        }
    }

    // Expected compressed_len, subsequent args are supplied to the variant's `compress` call.
    #[cfg(feature = "snappy")]
    test_variant!(snappy, header = b"\xff\x06\x00\x00\x73\x4e\x61\x50\x70\x59");

    #[cfg(feature = "gzip")]
    test_variant!(gzip, header = b"\x1f\x8b\x08\x00\x00", None);

    #[cfg(feature = "brotli")]
    test_variant!(brotli, header = b"\xcb\xff", None);

    #[cfg(feature = "bzip2")]
    test_variant!(bzip2, header = b"BZh6", None);

    #[cfg(feature = "deflate")]
    test_variant!(deflate, header = b"\xec\xcb\xcb\x09", None);

    #[cfg(feature = "zstd")]
    test_variant!(zstd, header = b"\x28\xb5\x2f\xfd", None);

    #[cfg(feature = "lz4")]
    test_variant!(lz4, header = b"\x04\x22\x4d\x18", None);

    #[cfg(feature = "blosc2")]
    test_variant!(blosc2, header = b"\x9e\xa8\x62\x32");

    #[cfg(feature = "xz")]
    #[allow(non_upper_case_globals)]
    const format: Option<crate::xz::Format> = None;

    #[allow(non_upper_case_globals)]
    #[cfg(feature = "xz")]
    const check: Option<crate::xz::Check> = None;

    #[allow(non_upper_case_globals)]
    #[cfg(feature = "xz")]
    const filters: Option<crate::xz::Filters> = None;

    #[allow(non_upper_case_globals)]
    #[cfg(feature = "xz")]
    const opts: Option<crate::xz::LzmaOptions> = None;

    #[cfg(feature = "xz")]
    test_variant!(
        xz,
        header = b"\xFD\x37\x7A\x58\x5A\x00",
        None,
        format,
        check,
        filters,
        opts
    );
}
