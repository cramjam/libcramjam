//! Static tables for DEFLATE (RFC 1951)

/// Base lengths for length codes 257..285
pub const LENGTH_BASE: [u16; 29] = [
    3, 4, 5, 6, 7, 8, 9, 10, // 257-264
    11, 13, 15, 17, // 265-268
    19, 23, 27, 31, // 269-272
    35, 43, 51, 59, // 273-276
    67, 83, 99, 115, // 277-280
    131, 163, 195, 227, // 281-284
    258, // 285
];

/// Extra bits for length codes 257..285
pub const LENGTH_EXTRA: [u8; 29] = [
    0, 0, 0, 0, 0, 0, 0, 0, // 257-264
    1, 1, 1, 1, // 265-268
    2, 2, 2, 2, // 269-272
    3, 3, 3, 3, // 273-276
    4, 4, 4, 4, // 277-280
    5, 5, 5, 5, // 281-284
    0, // 285
];

/// Base distances for distance codes 0..29
pub const DISTANCE_BASE: [u16; 30] = [
    1, 2, 3, 4, 5, 7, 9, 13, 17, 25, 33, 49, 65, 97, 129, 193, 257, 385, 513, 769, 1025, 1537,
    2049, 3073, 4097, 6145, 8193, 12289, 16385, 24577,
];

/// Extra bits for distance codes 0..29
pub const DISTANCE_EXTRA: [u8; 30] = [
    0, 0, 0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 8, 8, 9, 9, 10, 10, 11, 11, 12, 12,
    13, 13,
];

/// Order of code length codes in dynamic block headers
pub const CODE_LENGTH_ORDER: [usize; 19] = [
    16, 17, 18, 0, 8, 7, 9, 6, 10, 5, 11, 4, 12, 3, 13, 2, 14, 1, 15,
];

/// Fixed Huffman code lengths for literal/length alphabet (RFC 1951 section 3.2.6)
pub fn fixed_literal_lengths() -> [u8; 288] {
    let mut lengths = [0u8; 288];
    let mut i = 0;
    while i <= 143 {
        lengths[i] = 8;
        i += 1;
    }
    while i <= 255 {
        lengths[i] = 9;
        i += 1;
    }
    while i <= 279 {
        lengths[i] = 7;
        i += 1;
    }
    while i <= 287 {
        lengths[i] = 8;
        i += 1;
    }
    lengths
}

/// Fixed distance code lengths: all 5 bits for 32 symbols
pub fn fixed_distance_lengths() -> [u8; 32] {
    [5u8; 32]
}

/// Map a match length (3..=258) to (symbol 257..285, extra_bits, extra_value)
pub fn length_to_symbol(length: u16) -> (u16, u8, u16) {
    for i in (0..29).rev() {
        if length >= LENGTH_BASE[i] {
            return (257 + i as u16, LENGTH_EXTRA[i], length - LENGTH_BASE[i]);
        }
    }
    unreachable!("invalid length: {}", length)
}

/// Map a distance (1..=32768) to (symbol 0..29, extra_bits, extra_value)
pub fn distance_to_symbol(dist: u16) -> (u8, u8, u16) {
    for i in (0..30).rev() {
        if dist >= DISTANCE_BASE[i] {
            return (i as u8, DISTANCE_EXTRA[i], dist - DISTANCE_BASE[i]);
        }
    }
    unreachable!("invalid distance: {}", dist)
}
