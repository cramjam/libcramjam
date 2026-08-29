//! Compression parameters per level — `ZSTD_defaultCParameters` from C zstd
//! 1.5.7 (`clevels.h`) plus `ZSTD_adjustCParams_internal`.

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
#[repr(u8)]
pub enum Strategy {
    Fast = 1,
    Dfast = 2,
    Greedy = 3,
    Lazy = 4,
    Lazy2 = 5,
}

#[derive(Clone, Copy, Debug)]
pub struct CParams {
    pub window_log: u32,
    pub chain_log: u32,
    pub hash_log: u32,
    pub search_log: u32,
    pub min_match: u32,
    pub target_length: u32,
    pub strategy: Strategy,
}

pub const MAX_CLEVEL: i32 = 22;

// (W, C, H, S, L, TL, strat) — levels 0..=22, four srcSize tiers.
type Row = (u8, u8, u8, u8, u8, u16, Strategy);

use Strategy::*;

/// Levels above 12 (btlazy2/btopt/btultra) are not implemented; they map to
/// the strongest lazy2 configuration of the tier, which is what C's own
/// table uses for the last lazy2 row.
const TABLE: [[Row; 23]; 4] = [
    // "default" — srcSize > 256 KB
    [
        (19, 12, 13, 1, 6, 1, Fast),
        (19, 13, 14, 1, 7, 0, Fast),
        (20, 15, 16, 1, 6, 0, Fast),
        (21, 16, 17, 1, 5, 0, Dfast),
        (21, 18, 18, 1, 5, 0, Dfast),
        (21, 18, 19, 3, 5, 2, Greedy),
        (21, 18, 19, 3, 5, 4, Lazy),
        (21, 19, 20, 4, 5, 8, Lazy),
        (21, 19, 20, 4, 5, 16, Lazy2),
        (22, 20, 21, 4, 5, 16, Lazy2),
        (22, 21, 22, 5, 5, 16, Lazy2),
        (22, 21, 22, 6, 5, 16, Lazy2),
        (22, 22, 23, 6, 5, 32, Lazy2),
        (22, 22, 23, 6, 5, 32, Lazy2),
        (22, 22, 23, 6, 5, 32, Lazy2),
        (22, 22, 23, 6, 5, 32, Lazy2),
        (22, 22, 23, 6, 5, 32, Lazy2),
        (22, 22, 23, 6, 5, 32, Lazy2),
        (22, 22, 23, 6, 5, 32, Lazy2),
        (22, 22, 23, 6, 5, 32, Lazy2),
        (22, 22, 23, 6, 5, 32, Lazy2),
        (22, 22, 23, 6, 5, 32, Lazy2),
        (22, 22, 23, 6, 5, 32, Lazy2),
    ],
    // srcSize <= 256 KB
    [
        (18, 12, 13, 1, 5, 1, Fast),
        (18, 13, 14, 1, 6, 0, Fast),
        (18, 14, 14, 1, 5, 0, Dfast),
        (18, 16, 16, 1, 4, 0, Dfast),
        (18, 16, 17, 3, 5, 2, Greedy),
        (18, 17, 18, 5, 5, 2, Greedy),
        (18, 18, 19, 3, 5, 4, Lazy),
        (18, 18, 19, 4, 4, 4, Lazy),
        (18, 18, 19, 4, 4, 8, Lazy2),
        (18, 18, 19, 5, 4, 8, Lazy2),
        (18, 18, 19, 6, 4, 8, Lazy2),
        (18, 18, 19, 6, 4, 8, Lazy2),
        (18, 18, 19, 6, 4, 8, Lazy2),
        (18, 18, 19, 6, 4, 8, Lazy2),
        (18, 18, 19, 6, 4, 8, Lazy2),
        (18, 18, 19, 6, 4, 8, Lazy2),
        (18, 18, 19, 6, 4, 8, Lazy2),
        (18, 18, 19, 6, 4, 8, Lazy2),
        (18, 18, 19, 6, 4, 8, Lazy2),
        (18, 18, 19, 6, 4, 8, Lazy2),
        (18, 18, 19, 6, 4, 8, Lazy2),
        (18, 18, 19, 6, 4, 8, Lazy2),
        (18, 18, 19, 6, 4, 8, Lazy2),
    ],
    // srcSize <= 128 KB
    [
        (17, 12, 12, 1, 5, 1, Fast),
        (17, 12, 13, 1, 6, 0, Fast),
        (17, 13, 15, 1, 5, 0, Fast),
        (17, 15, 16, 2, 5, 0, Dfast),
        (17, 17, 17, 2, 4, 0, Dfast),
        (17, 16, 17, 3, 4, 2, Greedy),
        (17, 16, 17, 3, 4, 4, Lazy),
        (17, 16, 17, 3, 4, 8, Lazy2),
        (17, 16, 17, 4, 4, 8, Lazy2),
        (17, 16, 17, 5, 4, 8, Lazy2),
        (17, 16, 17, 6, 4, 8, Lazy2),
        (17, 16, 17, 6, 4, 8, Lazy2),
        (17, 16, 17, 6, 4, 8, Lazy2),
        (17, 16, 17, 6, 4, 8, Lazy2),
        (17, 16, 17, 6, 4, 8, Lazy2),
        (17, 16, 17, 6, 4, 8, Lazy2),
        (17, 16, 17, 6, 4, 8, Lazy2),
        (17, 16, 17, 6, 4, 8, Lazy2),
        (17, 16, 17, 6, 4, 8, Lazy2),
        (17, 16, 17, 6, 4, 8, Lazy2),
        (17, 16, 17, 6, 4, 8, Lazy2),
        (17, 16, 17, 6, 4, 8, Lazy2),
        (17, 16, 17, 6, 4, 8, Lazy2),
    ],
    // srcSize <= 16 KB
    [
        (14, 12, 13, 1, 5, 1, Fast),
        (14, 14, 15, 1, 5, 0, Fast),
        (14, 14, 15, 1, 4, 0, Fast),
        (14, 14, 15, 2, 4, 0, Dfast),
        (14, 14, 14, 4, 4, 2, Greedy),
        (14, 14, 14, 3, 4, 4, Lazy),
        (14, 14, 14, 4, 4, 8, Lazy2),
        (14, 14, 14, 6, 4, 8, Lazy2),
        (14, 14, 14, 8, 4, 8, Lazy2),
        (14, 14, 14, 8, 4, 8, Lazy2),
        (14, 14, 14, 8, 4, 8, Lazy2),
        (14, 14, 14, 8, 4, 8, Lazy2),
        (14, 14, 14, 8, 4, 8, Lazy2),
        (14, 14, 14, 8, 4, 8, Lazy2),
        (14, 14, 14, 8, 4, 8, Lazy2),
        (14, 14, 14, 8, 4, 8, Lazy2),
        (14, 14, 14, 8, 4, 8, Lazy2),
        (14, 14, 14, 8, 4, 8, Lazy2),
        (14, 14, 14, 8, 4, 8, Lazy2),
        (14, 14, 14, 8, 4, 8, Lazy2),
        (14, 14, 14, 8, 4, 8, Lazy2),
        (14, 14, 14, 8, 4, 8, Lazy2),
        (14, 14, 14, 8, 4, 8, Lazy2),
    ],
];

fn highbit32(v: u32) -> u32 {
    debug_assert!(v != 0);
    31 - v.leading_zeros()
}

/// `ZSTD_getCParams` + `ZSTD_adjustCParams_internal` for a known source size.
pub fn cparams(level: i32, src_size: usize) -> CParams {
    let level = level.clamp(1, MAX_CLEVEL) as usize;
    let tier = if src_size > 256 * 1024 {
        0
    } else if src_size > 128 * 1024 {
        1
    } else if src_size > 16 * 1024 {
        2
    } else {
        3
    };
    let (w, c, h, s, l, tl, strat) = TABLE[tier][level];
    let mut p = CParams {
        window_log: w as u32,
        chain_log: c as u32,
        hash_log: h as u32,
        search_log: s as u32,
        min_match: l as u32,
        target_length: tl as u32,
        strategy: strat,
    };
    // ZSTD_adjustCParams_internal: shrink the window to the source, then
    // keep the tables proportionate.
    const MIN_WINDOW_LOG: u32 = 10;
    if src_size > 0 {
        let src_log = if src_size <= 1 { 1 } else { highbit32((src_size - 1) as u32) + 1 };
        let src_log = src_log.max(MIN_WINDOW_LOG);
        if p.window_log > src_log {
            p.window_log = src_log;
        }
    }
    let cycle_log = p.chain_log; // no bt strategies here
    if cycle_log > p.window_log {
        p.chain_log -= cycle_log - p.window_log;
    }
    if p.hash_log > p.window_log + 1 {
        p.hash_log = p.window_log + 1;
    }
    if p.window_log < MIN_WINDOW_LOG {
        p.window_log = MIN_WINDOW_LOG;
    }
    p
}
