// IVF + f32 index loader and KNN search (V3).
//
// File format documented in tools/preprocess/src/main.rs.
//
// Query flow:
//   1. Compute L2² in f32 between query and each centroid → pick nprobe nearest.
//   2. For each chosen cell, exact f32 L2² scan with AVX2 (FMA).
//   3. Maintain a 5-slot top-K, then count fraud labels.

use crate::vector::{Query, D};
use memmap2::Mmap;
use std::fs::File;

const MAGIC_V3: &[u8; 8] = b"RINHAV03";
const VEC_STRIDE: usize = 16; // 16 f32 per vector (64 bytes)

pub struct Index {
    pub n: usize,
    pub d: usize,
    pub n_cells: usize,
    pub nprobe: usize,

    _mmap: Mmap,
    centroids: *const f32,        // [n_cells * D]
    cell_offset: *const u32,      // [n_cells + 1]
    vectors_f32: *const f32,      // [n * VEC_STRIDE]
    labels: *const u8,            // [n]
}

unsafe impl Send for Index {}
unsafe impl Sync for Index {}

impl Index {
    pub fn open(path: &str) -> std::io::Result<Self> {
        let f = File::open(path)?;
        let mmap = unsafe { Mmap::map(&f)? };
        if mmap.len() < 40 || &mmap[..8] != MAGIC_V3 {
            return Err(std::io::Error::new(std::io::ErrorKind::InvalidData, "bad magic v3"));
        }
        let n        = u64::from_le_bytes(mmap[8..16].try_into().unwrap()) as usize;
        let d        = u64::from_le_bytes(mmap[16..24].try_into().unwrap()) as usize;
        let n_cells  = u64::from_le_bytes(mmap[24..32].try_into().unwrap()) as usize;
        let mut nprobe = u64::from_le_bytes(mmap[32..40].try_into().unwrap()) as usize;
        if let Ok(env_np) = std::env::var("NPROBE") {
            if let Ok(v) = env_np.parse::<usize>() { nprobe = v; }
        }
        if d != D {
            return Err(std::io::Error::new(std::io::ErrorKind::InvalidData, "bad dim"));
        }
        let mut off = 40usize;
        let centroids = mmap[off..].as_ptr() as *const f32;
        off += n_cells * D * 4;
        let cell_offset = mmap[off..].as_ptr() as *const u32;
        off += (n_cells + 1) * 4;
        let vectors_f32 = mmap[off..].as_ptr() as *const f32;
        off += n * VEC_STRIDE * 4;
        let labels = mmap[off..].as_ptr();
        off += n;
        if mmap.len() < off {
            return Err(std::io::Error::new(std::io::ErrorKind::InvalidData, "truncated"));
        }

        // Pre-fault all pages so the first requests don't pay first-touch cost.
        let _ = mmap.advise(memmap2::Advice::WillNeed);
        let mut sum: u64 = 0;
        let bytes: &[u8] = mmap.as_ref();
        let mut i = 0;
        while i < bytes.len() {
            sum = sum.wrapping_add(bytes[i] as u64);
            i += 4096;
        }
        std::hint::black_box(sum);

        let idx = Index {
            n, d, n_cells, nprobe,
            _mmap: mmap, centroids, cell_offset, vectors_f32, labels,
        };

        // Self-warm: run synthetic queries that exercise the KNN code path on
        // every cell. Without this, the Rinha k6 ramp pays cold-cache cost on
        // the first ~1000 requests and the p99 tail explodes (we observed
        // 405ms → 212ms locally just by ab-warming before k6).
        let n_warm = std::env::var("WARMUP_QUERIES")
            .ok().and_then(|s| s.parse().ok()).unwrap_or(2000usize);
        idx.self_warm(n_warm);
        eprintln!("[index] self-warm: {} queries done", n_warm);

        Ok(idx)
    }

    fn self_warm(&self, n_queries: usize) {
        let mut q = Query::default();
        let mut state: u32 = 0xDEADBEEFu32;
        for _ in 0..n_queries {
            for k in 0..D {
                state = state.wrapping_mul(1664525).wrapping_add(1013904223);
                let f = ((state >> 8) & 0xFFFFFF) as f32 / 16_777_216.0;
                q.v[k] = if k == 5 || k == 6 {
                    if (state & 1) == 0 { -1.0 } else { f }
                } else {
                    f
                };
            }
            std::hint::black_box(self.knn5_count_frauds(&q));
        }
    }

    /// Pad the query to 16 floats (same layout as references).
    fn pad_query(q: &Query) -> [f32; VEC_STRIDE] {
        let mut out = [0f32; VEC_STRIDE];
        out[..D].copy_from_slice(&q.v);
        out
    }

    pub fn knn5_count_frauds(&self, q: &Query) -> u32 {
        let qpad = Self::pad_query(q);

        // 1. Distance to each centroid (f32), pick nprobe nearest via partial sort.
        //    Stack-allocated buffer — no heap alloc on the hot path.
        const MAX_CELLS: usize = 4096;
        let n_cells = self.n_cells.min(MAX_CELLS);
        let mut cell_d: [(f32, u32); MAX_CELLS] = [(f32::INFINITY, 0u32); MAX_CELLS];
        unsafe {
            for c in 0..n_cells {
                let base = self.centroids.add(c * D);
                let mut acc = 0f32;
                for k in 0..D {
                    let diff = *base.add(k) - q.v[k];
                    acc += diff * diff;
                }
                cell_d[c] = (acc, c as u32);
            }
        }
        let np = self.nprobe.min(n_cells);
        cell_d[..n_cells].select_nth_unstable_by(np - 1, |a, b|
            a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));

        // 2. Top-5 over chosen cells, f32 L2².
        //    Hoisted max_d / max_idx: only re-scan the 5 slots when we actually
        //    insert (which happens at most ~5 times per query in steady state).
        let mut dists = [f32::INFINITY; 5];
        let mut labs = [0u8; 5];
        let mut max_d = f32::INFINITY;
        let mut max_idx = 0usize;

        unsafe {
            let co = self.cell_offset;
            for slot in 0..np {
                let cell = cell_d[slot].1 as usize;
                let start = *co.add(cell) as usize;
                let end   = *co.add(cell + 1) as usize;
                for i in start..end {
                    let base = self.vectors_f32.add(i * VEC_STRIDE);
                    let d2 = l2sq_f32_16(base, qpad.as_ptr());
                    if d2 < max_d {
                        dists[max_idx] = d2;
                        labs[max_idx] = *self.labels.add(i);
                        // Recompute max — runs O(slot fills) total, ~5 times per query.
                        max_d = dists[0]; max_idx = 0;
                        if dists[1] > max_d { max_d = dists[1]; max_idx = 1; }
                        if dists[2] > max_d { max_d = dists[2]; max_idx = 2; }
                        if dists[3] > max_d { max_d = dists[3]; max_idx = 3; }
                        if dists[4] > max_d { max_d = dists[4]; max_idx = 4; }
                    }
                }
            }
        }

        labs.iter().map(|&l| l as u32).sum::<u32>()
    }
}

// Both `a` and `b` point to 16 contiguous floats; the last 2 are 0 (padding).
// Runtime dispatch — the cfg-gated path was being excluded when RUSTFLAGS did
// not propagate to the cfg layer, leaving us with a scalar build silently.
#[inline(always)]
unsafe fn l2sq_f32_16(a: *const f32, b: *const f32) -> f32 {
    #[cfg(target_arch = "x86_64")]
    {
        if std::arch::is_x86_feature_detected!("avx2") && std::arch::is_x86_feature_detected!("fma") {
            return l2sq_f32_16_avx2(a, b);
        }
    }
    let mut acc = 0f32;
    for k in 0..D {
        let d = *a.add(k) - *b.add(k);
        acc += d * d;
    }
    acc
}

#[cfg(target_arch = "x86_64")]
#[inline]
#[target_feature(enable = "avx2,fma")]
unsafe fn l2sq_f32_16_avx2(a: *const f32, b: *const f32) -> f32 {
    use std::arch::x86_64::*;
    let a0 = _mm256_loadu_ps(a);
    let a1 = _mm256_loadu_ps(a.add(8));
    let b0 = _mm256_loadu_ps(b);
    let b1 = _mm256_loadu_ps(b.add(8));
    let d0 = _mm256_sub_ps(a0, b0);
    let d1 = _mm256_sub_ps(a1, b1);
    // FMA: acc = d*d + acc, starting from 0
    let zero = _mm256_setzero_ps();
    let s0 = _mm256_fmadd_ps(d0, d0, zero);
    let s = _mm256_fmadd_ps(d1, d1, s0);
    // horizontal sum of 8 floats
    let lo = _mm256_castps256_ps128(s);
    let hi = _mm256_extractf128_ps(s, 1);
    let s = _mm_add_ps(lo, hi);
    let s = _mm_hadd_ps(s, s);
    let s = _mm_hadd_ps(s, s);
    _mm_cvtss_f32(s)
}
