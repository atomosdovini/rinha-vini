// Offline preprocessing for Rinha de Backend 2026.
//
// V3 artifact (magic RINHAV03): IVF + raw f32 vectors.
//
// We tried int8 quantization (V2) — it caused ~0.25 % decision flips on
// boundary cases where the 5th-vs-6th nearest neighbour swapped due to ~0.015
// L2² quant noise. Switched to f32 to match the ground-truth labels exactly.
// Total footprint stays under 350 MB because the index is mmap-shared between
// both replicas.
//
// Layout (little-endian):
//   magic        [u8;   8] = "RINHAV03"
//   n            u64
//   d            u64                  # always 14
//   n_cells      u64                  # 1024
//   default_np   u64                  # default nprobe
//   centroids    [f32;  n_cells * D]
//   cell_offset  [u32;  n_cells + 1]
//   vectors_f32  [f32;  n * VEC_STRIDE]   # padded to 16 floats per vector for SIMD
//   labels       [u8;   n]                # 0 legit, 1 fraud

use std::env;
use std::fs::File;
use std::io::{BufReader, BufWriter, Read, Write};

use flate2::read::GzDecoder;

const D: usize = 14;
const VEC_STRIDE: usize = 16; // 16 f32 = 64 bytes, AVX2-friendly
const MAGIC: &[u8; 8] = b"RINHAV03";

const N_CELLS: usize = 1024;
const KMEANS_ITERS: usize = 12;
const KMEANS_SAMPLE: usize = 200_000;
const DEFAULT_NPROBE: u64 = 24;

fn main() -> std::io::Result<()> {
    let args: Vec<String> = env::args().collect();
    let in_path = args.get(1).cloned().unwrap_or_else(|| "/work/references.json.gz".into());
    let out_path = args.get(2).cloned().unwrap_or_else(|| "/opt/index.bin".into());

    eprintln!("[preprocess] in={} out={}", in_path, out_path);

    // ── 1. decompress + parse ────────────────────────────────────────────────
    let mut buf = Vec::with_capacity(1 << 28);
    {
        let f = File::open(&in_path)?;
        let mut gz = GzDecoder::new(BufReader::new(f));
        gz.read_to_end(&mut buf)?;
    }
    eprintln!("[preprocess] decompressed {} bytes", buf.len());

    let mut vectors: Vec<f32> = Vec::with_capacity(3_000_000 * D);
    let mut labels: Vec<u8> = Vec::with_capacity(3_000_000);

    let mut i = 0usize;
    let mut count = 0usize;
    while i < buf.len() {
        while i < buf.len() && buf[i] != b'{' { i += 1; }
        if i >= buf.len() { break; }
        let obj_start = i;
        let mut j = i;
        let mut depth = 0i32;
        let mut end = i;
        while j < buf.len() {
            match buf[j] {
                b'{' => depth += 1,
                b'}' => { depth -= 1; if depth == 0 { end = j; break; } }
                _ => {}
            }
            j += 1;
        }
        if end == obj_start { break; }
        let obj = &buf[obj_start..=end];
        let vstart = find_subseq(obj, b"\"vector\"").and_then(|p| {
            let mut k = p + b"\"vector\"".len();
            while k < obj.len() && obj[k] != b'[' { k += 1; }
            if k < obj.len() { Some(k + 1) } else { None }
        });
        let lstart = find_subseq(obj, b"\"label\"");
        if let (Some(vs), Some(ls)) = (vstart, lstart) {
            let mut k = vs;
            let mut got = [0f32; D];
            for slot in 0..D {
                while k < obj.len() && (obj[k] == b' ' || obj[k] == b',') { k += 1; }
                let start = k;
                while k < obj.len() && obj[k] != b',' && obj[k] != b']' { k += 1; }
                let s = std::str::from_utf8(&obj[start..k]).unwrap_or("0").trim();
                got[slot] = s.parse::<f32>().unwrap_or(0.0);
            }
            let mut k = ls + b"\"label\"".len();
            while k < obj.len() && obj[k] != b'"' { k += 1; }
            let s2 = k + 1;
            let mut k2 = s2;
            while k2 < obj.len() && obj[k2] != b'"' { k2 += 1; }
            let lab = if &obj[s2..k2] == b"fraud" { 1u8 } else { 0u8 };
            vectors.extend_from_slice(&got);
            labels.push(lab);
            count += 1;
            if count % 500_000 == 0 {
                eprintln!("[preprocess] parsed {} records", count);
            }
        }
        i = end + 1;
    }
    let n = count;
    eprintln!("[preprocess] parsed {} vectors", n);

    // ── 2. k-means (mini-batch on a sample) ─────────────────────────────────
    eprintln!("[preprocess] k-means: {} cells, {} iters, sample {}", N_CELLS, KMEANS_ITERS, KMEANS_SAMPLE);
    let mut centroids = init_centroids(&vectors, n, N_CELLS);
    for it in 0..KMEANS_ITERS {
        let inertia = kmeans_iter(&vectors, n, &mut centroids, KMEANS_SAMPLE);
        eprintln!("[preprocess]   iter {}: inertia/sample = {:.4}", it, inertia);
    }

    // ── 3. assign every vector to its nearest centroid ──────────────────────
    eprintln!("[preprocess] assigning {} vectors to cells...", n);
    let mut assign: Vec<u32> = vec![0; n];
    for i in 0..n {
        let v = &vectors[i * D..(i + 1) * D];
        assign[i] = nearest_centroid(v, &centroids) as u32;
    }

    // ── 4. sort by cell id ──────────────────────────────────────────────────
    eprintln!("[preprocess] sorting by cell...");
    let mut order: Vec<u32> = (0..n as u32).collect();
    order.sort_unstable_by_key(|&i| assign[i as usize]);

    // ── 5. emit padded f32 layout ──────────────────────────────────────────
    let mut vec_f32: Vec<f32> = Vec::with_capacity(n * VEC_STRIDE);
    let mut lab_sorted: Vec<u8> = Vec::with_capacity(n);
    let mut cell_offset: Vec<u32> = vec![0; N_CELLS + 1];

    let mut last_cell = 0u32;
    for (pos, &i) in order.iter().enumerate() {
        let cell = assign[i as usize];
        while last_cell < cell {
            last_cell += 1;
            cell_offset[last_cell as usize] = pos as u32;
        }
        let v = &vectors[i as usize * D..(i as usize + 1) * D];
        let mut row = [0f32; VEC_STRIDE];
        row[..D].copy_from_slice(v);
        vec_f32.extend_from_slice(&row);
        lab_sorted.push(labels[i as usize]);
    }
    while (last_cell as usize) < N_CELLS {
        last_cell += 1;
        cell_offset[last_cell as usize] = n as u32;
    }
    cell_offset[N_CELLS] = n as u32;

    // ── 6. write artifact ───────────────────────────────────────────────────
    let f = File::create(&out_path)?;
    let mut w = BufWriter::with_capacity(1 << 20, f);
    w.write_all(MAGIC)?;
    w.write_all(&(n as u64).to_le_bytes())?;
    w.write_all(&(D as u64).to_le_bytes())?;
    w.write_all(&(N_CELLS as u64).to_le_bytes())?;
    w.write_all(&DEFAULT_NPROBE.to_le_bytes())?;
    // centroids
    let centroids_bytes = unsafe {
        std::slice::from_raw_parts(centroids.as_ptr() as *const u8, centroids.len() * 4)
    };
    w.write_all(centroids_bytes)?;
    // cell offsets
    let co_bytes = unsafe {
        std::slice::from_raw_parts(cell_offset.as_ptr() as *const u8, cell_offset.len() * 4)
    };
    w.write_all(co_bytes)?;
    // vectors (padded to 16 floats each = 64 B)
    let v_bytes = unsafe {
        std::slice::from_raw_parts(vec_f32.as_ptr() as *const u8, vec_f32.len() * 4)
    };
    w.write_all(v_bytes)?;
    // labels
    w.write_all(&lab_sorted)?;
    w.flush()?;

    let total = 40 + centroids.len() * 4 + cell_offset.len() * 4 + vec_f32.len() * 4 + lab_sorted.len();
    eprintln!("[preprocess] wrote {} ({} bytes)", out_path, total);
    Ok(())
}

// ──────────────────────────────────────────────────────────────────────────────
// k-means helpers
// ──────────────────────────────────────────────────────────────────────────────

fn init_centroids(vectors: &[f32], n: usize, k: usize) -> Vec<f32> {
    // Simple: pick k vectors evenly spaced as initial centroids.
    let mut out = Vec::with_capacity(k * D);
    let stride = (n / k).max(1);
    for c in 0..k {
        let idx = (c * stride) % n;
        out.extend_from_slice(&vectors[idx * D..(idx + 1) * D]);
    }
    out
}

fn nearest_centroid(v: &[f32], centroids: &[f32]) -> usize {
    let mut best = 0usize;
    let mut best_d = f32::INFINITY;
    let k = centroids.len() / D;
    for c in 0..k {
        let base = c * D;
        let mut acc = 0f32;
        for kk in 0..D {
            let diff = v[kk] - centroids[base + kk];
            acc += diff * diff;
        }
        if acc < best_d {
            best_d = acc;
            best = c;
        }
    }
    best
}

fn kmeans_iter(vectors: &[f32], n: usize, centroids: &mut [f32], sample: usize) -> f64 {
    let k = centroids.len() / D;
    let mut sums = vec![0f32; k * D];
    let mut counts = vec![0u32; k];
    let mut total = 0f64;
    let step = (n / sample).max(1);
    let mut samples = 0usize;
    let mut idx = 0usize;
    while idx < n && samples < sample {
        let v = &vectors[idx * D..(idx + 1) * D];
        // nearest
        let mut best = 0usize;
        let mut best_d = f32::INFINITY;
        for c in 0..k {
            let base = c * D;
            let mut acc = 0f32;
            for kk in 0..D {
                let diff = v[kk] - centroids[base + kk];
                acc += diff * diff;
            }
            if acc < best_d {
                best_d = acc;
                best = c;
            }
        }
        total += best_d as f64;
        let bb = best * D;
        for kk in 0..D {
            sums[bb + kk] += v[kk];
        }
        counts[best] += 1;
        samples += 1;
        idx += step;
    }
    for c in 0..k {
        if counts[c] > 0 {
            let inv = 1.0 / counts[c] as f32;
            let cb = c * D;
            for kk in 0..D {
                centroids[cb + kk] = sums[cb + kk] * inv;
            }
        }
    }
    total / samples as f64
}

fn find_subseq(haystack: &[u8], needle: &[u8]) -> Option<usize> {
    if needle.is_empty() || haystack.len() < needle.len() { return None; }
    for i in 0..=haystack.len() - needle.len() {
        if &haystack[i..i + needle.len()] == needle { return Some(i); }
    }
    None
}
