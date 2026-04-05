//! Adler-32 as specified in RFC 1950
//! BASE = 65521 (largest prime < 65536)

const BASE: u32 = 65521;
/// Largest n such that 255*n*(n+1)/2 + (n+1)*(BASE-1) fits in a u32.
const NMAX: usize = 5552;

/// Compute Adler-32 of a byte slice.
pub fn adler32(data: &[u8]) -> u32 {
    let mut s1: u32 = 1;
    let mut s2: u32 = 0;

    for chunk in data.chunks(NMAX) {
        for &byte in chunk {
            s1 += byte as u32;
            s2 += s1;
        }
        s1 %= BASE;
        s2 %= BASE;
    }

    (s2 << 16) | s1
}

/// Incremental Adler-32 computation.
pub struct Adler32 {
    s1: u32,
    s2: u32,
    count: usize,
}

impl Adler32 {
    pub fn new() -> Self {
        Self {
            s1: 1,
            s2: 0,
            count: 0,
        }
    }

    pub fn update(&mut self, data: &[u8]) {
        for &byte in data {
            self.s1 += byte as u32;
            self.s2 += self.s1;
            self.count += 1;
            if self.count >= NMAX {
                self.s1 %= BASE;
                self.s2 %= BASE;
                self.count = 0;
            }
        }
    }

    pub fn finalize(mut self) -> u32 {
        self.s1 %= BASE;
        self.s2 %= BASE;
        (self.s2 << 16) | self.s1
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_adler32_empty() {
        assert_eq!(adler32(&[]), 1);
    }

    #[test]
    fn test_adler32_known() {
        // "Wikipedia" has Adler-32 = 0x11E60398
        assert_eq!(adler32(b"Wikipedia"), 0x11E6_0398);
    }

    #[test]
    fn test_adler32_incremental() {
        let data = b"Hello, world!";
        let expected = adler32(data);
        let mut a = Adler32::new();
        a.update(&data[..5]);
        a.update(&data[5..]);
        assert_eq!(a.finalize(), expected);
    }
}
