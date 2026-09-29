//! Branch / Call / Jump (BCJ) "simple" filters.
//!
//! BCJ filters preprocess executable code so that branch / call / jump
//! instructions encode their target as an absolute offset rather than a
//! position-relative offset.  Absolute offsets are far more compressible
//! than relative ones because identical call sites in different files
//! produce the same bytes.
//!
//! Each filter pair (encode/decode) is its own inverse: the encoder
//! converts relative→absolute, the decoder converts absolute→relative.
//!
//! All implementations are direct one-shot ports of liblzma's
//! `src/liblzma/simple/*.c`.  None of them allocate.  They all operate
//! in place on a `&mut [u8]` and return the number of bytes that were
//! actually transformed (may be smaller than `buf.len()` when the tail
//! of the buffer doesn't fit a full instruction).
//!
//! `now_pos` is the absolute byte offset of `buf[0]` within the original
//! stream — for one-shot block decode this is the BCJ filter's
//! `start_offset` property (or 0 if not present).

// The filter ids live with the shared option types (also used by the C backend).
pub use super::options::{FILTER_ARM, FILTER_ARMTHUMB, FILTER_IA64, FILTER_POWERPC, FILTER_SPARC, FILTER_X86};

/// True iff `id` is a BCJ filter id.
pub fn is_bcj(id: u64) -> bool {
    matches!(
        id,
        FILTER_X86
            | FILTER_POWERPC
            | FILTER_IA64
            | FILTER_ARM
            | FILTER_ARMTHUMB
            | FILTER_SPARC
    )
}

/// Apply the BCJ filter identified by `filter_id` to `buf` in place.
/// Returns the number of bytes processed (the tail beyond this is left
/// untouched, since it didn't fit a full instruction).
///
/// `is_encoder=false` runs the decode direction (absolute → relative),
/// which is what `decode_xz_stream` calls after running LZMA2.
pub fn apply(filter_id: u64, buf: &mut [u8], now_pos: u32, is_encoder: bool) -> usize {
    match filter_id {
        FILTER_X86 => x86_code(buf, now_pos, is_encoder),
        FILTER_POWERPC => powerpc_code(buf, now_pos, is_encoder),
        FILTER_IA64 => ia64_code(buf, now_pos, is_encoder),
        FILTER_ARM => arm_code(buf, now_pos, is_encoder),
        FILTER_ARMTHUMB => armthumb_code(buf, now_pos, is_encoder),
        FILTER_SPARC => sparc_code(buf, now_pos, is_encoder),
        _ => 0,
    }
}

// =========================================================================
// ARM (32-bit) — `BL` (Branch with Link) instructions
// =========================================================================
// Translated from `liblzma/simple/arm.c`.

fn arm_code(buf: &mut [u8], now_pos: u32, is_encoder: bool) -> usize {
    let mut i = 0;
    while i + 4 <= buf.len() {
        if buf[i + 3] == 0xEB {
            let mut src = (buf[i + 2] as u32) << 16
                | (buf[i + 1] as u32) << 8
                | (buf[i] as u32);
            src <<= 2;
            let pc = now_pos.wrapping_add(i as u32).wrapping_add(8);
            let dest = if is_encoder {
                pc.wrapping_add(src)
            } else {
                src.wrapping_sub(pc)
            } >> 2;
            buf[i + 2] = (dest >> 16) as u8;
            buf[i + 1] = (dest >> 8) as u8;
            buf[i] = dest as u8;
        }
        i += 4;
    }
    i
}

// =========================================================================
// ARM-Thumb — 32-bit `BL` instruction split across two 16-bit halfwords
// =========================================================================
// Translated from `liblzma/simple/armthumb.c`.

fn armthumb_code(buf: &mut [u8], now_pos: u32, is_encoder: bool) -> usize {
    let mut i = 0;
    while i + 4 <= buf.len() {
        if (buf[i + 1] & 0xF8) == 0xF0 && (buf[i + 3] & 0xF8) == 0xF8 {
            let src = ((buf[i + 1] as u32 & 7) << 19)
                | ((buf[i] as u32) << 11)
                | ((buf[i + 3] as u32 & 7) << 8)
                | (buf[i + 2] as u32);
            let src = src << 1;
            let pc = now_pos.wrapping_add(i as u32).wrapping_add(4);
            let dest = if is_encoder {
                pc.wrapping_add(src)
            } else {
                src.wrapping_sub(pc)
            } >> 1;
            buf[i + 1] = 0xF0 | ((dest >> 19) & 0x7) as u8;
            buf[i] = (dest >> 11) as u8;
            buf[i + 3] = 0xF8 | ((dest >> 8) & 0x7) as u8;
            buf[i + 2] = dest as u8;
            i += 2;
        }
        i += 2;
    }
    i
}

// =========================================================================
// PowerPC (big endian) — branch instruction (opcode 18)
// =========================================================================
// Translated from `liblzma/simple/powerpc.c`.

fn powerpc_code(buf: &mut [u8], now_pos: u32, is_encoder: bool) -> usize {
    let mut i = 0;
    while i + 4 <= buf.len() {
        // Branch (opcode 18 = 0x12), with the LK bit set (bit 0).
        if (buf[i] >> 2) == 0x12 && (buf[i + 3] & 3) == 1 {
            let src = ((buf[i] as u32 & 3) << 24)
                | ((buf[i + 1] as u32) << 16)
                | ((buf[i + 2] as u32) << 8)
                | (buf[i + 3] as u32 & !3u32);
            let pc = now_pos.wrapping_add(i as u32);
            let dest = if is_encoder {
                pc.wrapping_add(src)
            } else {
                src.wrapping_sub(pc)
            };
            buf[i] = 0x48 | ((dest >> 24) & 0x03) as u8;
            buf[i + 1] = (dest >> 16) as u8;
            buf[i + 2] = (dest >> 8) as u8;
            buf[i + 3] = (buf[i + 3] & 0x03) | (dest as u8 & !0x03u8);
        }
        i += 4;
    }
    i
}

// =========================================================================
// SPARC — `CALL` (opcode 0x40 ..) and `Bicc` (opcode 0x7F ..)
// =========================================================================
// Translated from `liblzma/simple/sparc.c`.

fn sparc_code(buf: &mut [u8], now_pos: u32, is_encoder: bool) -> usize {
    let mut i = 0;
    while i + 4 <= buf.len() {
        if (buf[i] == 0x40 && (buf[i + 1] & 0xC0) == 0x00)
            || (buf[i] == 0x7F && (buf[i + 1] & 0xC0) == 0xC0)
        {
            let src = ((buf[i] as u32) << 24)
                | ((buf[i + 1] as u32) << 16)
                | ((buf[i + 2] as u32) << 8)
                | (buf[i + 3] as u32);
            let src = src << 2;
            let pc = now_pos.wrapping_add(i as u32);
            let dest_base = if is_encoder {
                pc.wrapping_add(src)
            } else {
                src.wrapping_sub(pc)
            } >> 2;
            // Re-pack the 22-bit signed displacement back into the SPARC
            // call/branch encoding.
            let dest = (((0u32.wrapping_sub((dest_base >> 22) & 1)) << 22) & 0x3FFFFFFF)
                | (dest_base & 0x3FFFFF)
                | 0x40000000;
            buf[i] = (dest >> 24) as u8;
            buf[i + 1] = (dest >> 16) as u8;
            buf[i + 2] = (dest >> 8) as u8;
            buf[i + 3] = dest as u8;
        }
        i += 4;
    }
    i
}

// =========================================================================
// IA-64 (Itanium) — bundle of three 41-bit instruction slots
// =========================================================================
// Translated from `liblzma/simple/ia64.c`.

fn ia64_code(buf: &mut [u8], now_pos: u32, is_encoder: bool) -> usize {
    static BRANCH_TABLE: [u32; 32] = [
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 4, 4, 6, 6, 0, 0, 7, 7, 4, 4, 0, 0, 4, 4, 0, 0,
    ];

    let mut i = 0;
    while i + 16 <= buf.len() {
        let instr_template = (buf[i] & 0x1F) as u32;
        let mask = BRANCH_TABLE[instr_template as usize];
        let mut bit_pos: u32 = 5;

        for slot in 0..3u32 {
            if (mask >> slot) & 1 == 0 {
                bit_pos += 41;
                continue;
            }

            let byte_pos = (bit_pos >> 3) as usize;
            let bit_res = bit_pos & 7;
            let mut instruction: u64 = 0;
            for j in 0..6 {
                instruction += (buf[i + j + byte_pos] as u64) << (8 * j);
            }
            let inst_norm = instruction >> bit_res;

            if ((inst_norm >> 37) & 0xF) == 0x5 && ((inst_norm >> 9) & 0x7) == 0 {
                let mut src = ((inst_norm >> 13) & 0xFFFFF) as u32;
                src |= (((inst_norm >> 36) & 1) as u32) << 20;
                let src = src << 4;
                let pc = now_pos.wrapping_add(i as u32);
                let dest = if is_encoder {
                    pc.wrapping_add(src)
                } else {
                    src.wrapping_sub(pc)
                } >> 4;

                let mut new_norm = inst_norm;
                new_norm &= !(0x8FFFFFu64 << 13);
                new_norm |= (dest as u64 & 0xFFFFF) << 13;
                new_norm |= (dest as u64 & 0x100000) << (36 - 20);

                let mut new_instr = instruction & ((1u64 << bit_res) - 1);
                new_instr |= new_norm << bit_res;
                for j in 0..6 {
                    buf[i + j + byte_pos] = (new_instr >> (8 * j)) as u8;
                }
            }

            bit_pos += 41;
        }
        i += 16;
    }
    i
}

// =========================================================================
// x86 — `CALL` (`E8`) and `JMP` (`E9`) with stateful look-back
// =========================================================================
// Translated from `liblzma/simple/x86.c`.  The original keeps a per-stream
// `prev_mask` / `prev_pos` state across chunks; for one-shot block decode
// we initialise them to the documented "fresh stream" values
// (`prev_mask = 0`, `prev_pos = (uint32_t)(-5)`) and run the algorithm
// over the whole buffer.

fn x86_test_msbyte(b: u8) -> bool {
    b == 0 || b == 0xFF
}

fn x86_code(buf: &mut [u8], now_pos: u32, is_encoder: bool) -> usize {
    const MASK_TO_ALLOWED_STATUS: [bool; 8] =
        [true, true, true, false, true, false, false, false];
    const MASK_TO_BIT_NUMBER: [u32; 8] = [0, 1, 2, 2, 3, 3, 3, 3];

    if buf.len() < 5 {
        return 0;
    }

    let mut prev_mask: u32 = 0;
    let mut prev_pos: u32 = 0u32.wrapping_sub(5);

    if now_pos.wrapping_sub(prev_pos) > 5 {
        prev_pos = now_pos.wrapping_sub(5);
    }

    let limit = buf.len() - 5;
    let mut buffer_pos: usize = 0;

    while buffer_pos <= limit {
        let b = buf[buffer_pos];
        if b != 0xE8 && b != 0xE9 {
            buffer_pos += 1;
            continue;
        }

        let offset = now_pos
            .wrapping_add(buffer_pos as u32)
            .wrapping_sub(prev_pos);
        prev_pos = now_pos.wrapping_add(buffer_pos as u32);

        if offset > 5 {
            prev_mask = 0;
        } else {
            for _ in 0..offset {
                prev_mask &= 0x77;
                prev_mask <<= 1;
            }
        }

        let b4 = buf[buffer_pos + 4];

        if x86_test_msbyte(b4)
            && MASK_TO_ALLOWED_STATUS[((prev_mask >> 1) & 0x7) as usize]
            && (prev_mask >> 1) < 0x10
        {
            let mut src = ((b4 as u32) << 24)
                | ((buf[buffer_pos + 3] as u32) << 16)
                | ((buf[buffer_pos + 2] as u32) << 8)
                | (buf[buffer_pos + 1] as u32);

            let dest;
            loop {
                let pc = now_pos
                    .wrapping_add(buffer_pos as u32)
                    .wrapping_add(5);
                let candidate = if is_encoder {
                    src.wrapping_add(pc)
                } else {
                    src.wrapping_sub(pc)
                };
                if prev_mask == 0 {
                    dest = candidate;
                    break;
                }
                let i_idx = MASK_TO_BIT_NUMBER[(prev_mask >> 1) as usize];
                let bb = (candidate >> (24 - i_idx * 8)) as u8;
                if !x86_test_msbyte(bb) {
                    dest = candidate;
                    break;
                }
                src = candidate ^ ((1u32 << (32 - i_idx * 8)) - 1);
            }

            buf[buffer_pos + 4] = (!(((dest >> 24) & 1).wrapping_sub(1))) as u8;
            buf[buffer_pos + 3] = (dest >> 16) as u8;
            buf[buffer_pos + 2] = (dest >> 8) as u8;
            buf[buffer_pos + 1] = dest as u8;
            buffer_pos += 5;
            prev_mask = 0;
        } else {
            buffer_pos += 1;
            prev_mask |= 1;
            if x86_test_msbyte(b4) {
                prev_mask |= 0x10;
            }
        }
    }

    buffer_pos
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Round-trip a buffer through encode → decode for every BCJ filter
    /// and check that we recover the original bytes.  This is a self-
    /// consistency check; the cross-impl tests in `tests/cross_impl_xz.rs`
    /// verify our results match liblzma's.
    fn roundtrip(filter_id: u64, mut buf: Vec<u8>) {
        let original = buf.clone();
        apply(filter_id, &mut buf, 0, true);
        apply(filter_id, &mut buf, 0, false);
        assert_eq!(buf, original, "filter id {:#x} did not round-trip", filter_id);
    }

    #[test]
    fn roundtrip_all_filters_random_data() {
        // Random data — most positions won't match the BCJ pattern, but
        // the few that do still need to round-trip cleanly.
        let mut s: u32 = 0xC0FFEE_42;
        let buf: Vec<u8> = (0..1024)
            .map(|_| {
                s ^= s << 13;
                s ^= s >> 17;
                s ^= s << 5;
                (s >> 16) as u8
            })
            .collect();
        for id in [
            FILTER_X86,
            FILTER_POWERPC,
            FILTER_IA64,
            FILTER_ARM,
            FILTER_ARMTHUMB,
            FILTER_SPARC,
        ] {
            roundtrip(id, buf.clone());
        }
    }
}

/// Run the BCJ encoders of `bcj` (chain order) over a copy of `input`, or
/// hand back `input` itself when there are none.
pub(crate) fn bcj_encode<'a>(input: &'a [u8], bcj: &[u64], scratch: &'a mut Vec<u8>) -> &'a [u8] {
    if bcj.is_empty() {
        return input;
    }
    scratch.clear();
    scratch.extend_from_slice(input);
    for &id in bcj {
        apply(id, scratch, 0, true);
    }
    scratch
}
