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
        ($name:ident($compress_output:ident -> $decompress_output:ident), variant=$variant:ident, $(, $args:ident)*) => {
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
        ($variant:ident $(, $args:tt)*) => {
         #[cfg(test)]
         mod $variant {
            use super::*;
            round_trip!(roundtrip_compress_via_slice_decompress_via_slice(Slice -> Slice), variant=$variant, $(, $args)* );
            round_trip!(roundtrip_compress_via_slice_decompress_via_vector(Slice -> Vector), variant=$variant, $(, $args)* );
            round_trip!(roundtrip_compress_via_vector_decompress_via_slice(Vector -> Slice), variant=$variant, $(, $args)* );
            round_trip!(roundtrip_compress_via_vector_decompress_via_vector(Vector -> Vector), variant=$variant, $(, $args)* );
         }
        }
    }

    // Expected compressed_len, subsequent args are supplied to the variant's `compress` call.
    #[cfg(feature = "snappy")]
    test_variant!(snappy);

    #[cfg(feature = "gzip")]
    test_variant!(gzip, None);

    #[cfg(feature = "brotli")]
    test_variant!(brotli, None);

    #[cfg(feature = "bzip2")]
    test_variant!(bzip2, None);

    #[cfg(feature = "deflate")]
    test_variant!(deflate, None);

    #[cfg(feature = "zstd")]
    test_variant!(zstd, None);

    #[cfg(feature = "lz4")]
    test_variant!(lz4, None);

    #[cfg(feature = "blosc2")]
    test_variant!(blosc2);

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
    test_variant!(xz, None, format, check, filters, opts);
}
