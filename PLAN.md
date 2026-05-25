# Rinha de Backend 2026 — Engineering Plan

A systems-engineering blueprint for an ultra-specialized fraud-detection inference engine. Target: top-of-leaderboard `final_score` (ceiling +6000).

---

## 1. Executive summary

The challenge is **not a backend**. It is a 1-CPU / 350 MB **vector-search inference appliance** wrapped in HTTP. Per request we must:

1. Parse a small JSON payload (~400 bytes).
2. Build a 14-dim float vector via a fixed normalization.
3. Find the 5 exact-or-near-exact L2 nearest neighbors among **3,000,000** labeled reference vectors.
4. Return `approved = (fraud_count_in_top5 < 3)` and `fraud_score = fraud_count/5`.

Scoring rewards p99 ≤ 1 ms (saturates at +3000) and near-zero FP/FN/HTTP errors (+3000). Both ceilings are reachable simultaneously only with a **highly specialized in-memory ANN structure**, sub-millisecond JSON parsing, and a stack with negligible HTTP overhead. Round-robin between 2 API instances on a 1-CPU box means each replica sees ~half traffic but **competes for the same physical core**, so duplicated structures are pure overhead — memory layout must be designed under an effective budget of ~140–160 MB per replica.

Strategy in one line: **mmap a shared, read-only, quantized, IVF-clustered vector store; route each query to a handful of cells; do an exact L2 scan with SIMD on int8 vectors; return via a hand-tuned HTTP stack.**

Target operating point: 100–300 µs p50, ≤1 ms p99, FP+FN+Err rate ≤ 0.3 %.

---

## 2. Key insights from the official docs

- **Labels are ground truth from exact brute-force K=5 / L2 / 14-D over the same 3 M references.** ANN recall directly determines FP/FN. Recall@5 of 99.5 %+ is the right target; anything aggressive (HNSW with low ef, IVF with nprobe=1) will tank `score_det`.
- **Threshold is 0.6 over K=5 → decision flips on the 3rd-vs-4th neighbor.** Many borderline payloads will have ties at distance level; a missing one of the top-5 neighbors only flips the answer when it crosses the 3-fraud boundary. This makes the problem **error-tolerant** to occasional swaps among same-label neighbors. Most recall loss costs nothing.
- **Indices 5, 6 carry sentinel `-1`** for `last_transaction: null`. This naturally segregates "no-history" vectors into a distant cluster in L2 — perfect for a **first-level binary partition**.
- **HTTP error weight = 5, FN = 3, FP = 1.** Never throw — on any internal failure, return `{approved:true, fraud_score:0.0}` (or true negative bias) as a fallback. FP/FN ≪ Err.
- **k6 ramps to 900 req/s for 120 s, total ~70 k requests.** Combined load on both replicas. With 1 CPU, that's ~1.1 ms of CPU budget per request *worst-case mean* — already at the score ceiling. Anything beyond minimal user-space work is fatal.
- **Test data uses `test-data.json`**, but final test will use different payloads — we cannot memoize answers by `id`. We *can* assume distribution matches the reference set statistics.
- **References do not change.** Everything is offline: build-time decompression, normalization-aware bucketization, quantization, layout — and ship the artifact as a binary blob baked into the image layer.

---

## 3. Real engineering constraints

| Constraint | Value | Implication |
|---|---|---|
| CPU total | 1.0 | Two API instances *share one core*. Pinning helps cache locality but increases contention. |
| RAM total | 350 MB | LB ≤ 8 MB, kernel/overhead ~30 MB; ~310 MB for API pair. |
| Replicas | ≥ 2 | Duplicated in-process state doubles. Need **shared mmap** to amortize. |
| Network | bridge | ~30–80 µs added per hop vs host; LB adds another. |
| Image | linux/amd64 | AVX2 baseline likely (Mac Mini 2014 = Haswell → AVX2 yes, **no AVX-512**). |
| Test host | Mac Mini Late 2014, 8 GB RAM, Ubuntu 24.04 | 4C/8T Haswell i7-4578U or similar. Cache: 32K L1d/core, 256K L2/core, 4 MB L3 shared. |
| Port | 9999 (LB) | Two extra hops in the request path. |

The **L3 cache size (~4 MB)** is the most under-appreciated constraint: any "hot" working-set above 4 MB triggers DRAM accesses (~80 ns each). The full 3 M × 14 × 4 B = **168 MB** float vector store does not fit in cache at all. Quantization + IVF clustering is the only path to working in-L3 for the hot loop.

---

## 4. Dataset analysis ideas (must run, not assume)

Before designing anything, **profile the dataset offline**:

1. **Distribution per dimension**: mean/std/percentiles. Identify dimensions with low entropy (e.g., `is_online`, `card_present`, `unknown_merchant` are 0/1; `mcc_risk` is a tiny discrete set of ~11 values).
2. **`-1` prevalence**: % of vectors with `last_transaction = null`. Likely a clear bimodal subpopulation.
3. **Fraud rate per cell** under candidate IVF/grid partitions. If most cells are 100% one label, the search is trivial in those cells.
4. **K=5 ground-truth radius distribution**: how far is the 5th NN typically? This sets the **search radius** used to prune cells.
5. **PCA / per-dim variance**: identifies which dims dominate L2. With only 14 dims and many binary, the *effective* dimensionality is probably ~6–8.
6. **Duplicate-prefix rate**: after quantization, how many vectors collapse to identical codes? This gives a **deduplication multiplier** for the search.
7. **Edge cases in `test-data.json`**: 5 k requests with known answers. Use them as a recall-tuning oracle (must not be *used* at runtime, but *can* be used to validate offline that ANN matches brute force).

Output of this phase: a `dataset_stats.json` artifact that drives every later decision (cell count, quantization scale, sentinel handling).

---

## 5. Most promising architecture candidates (ranked)

### A. **Shared-memory IVF + int8 SIMD scan** (recommended baseline → final)
- Offline k-means on 14-dim float, ~256–1024 cells (auto-tuned).
- Vectors quantized to int8 per-dim (with a scalar dequant table) → 14 bytes/vec + 1 byte label = **15 bytes/vec × 3 M = 45 MB**.
- Sorted by cell, cell offsets in a tiny header; **mmap shared between both replicas**.
- Query: normalize → predict cell + nprobe neighbors (8–16 cells) → linear int8 L2 with AVX2 → top-5 heap.
- Expected: ~5–30 k vectors scanned per query, ~50–150 µs.

### B. **Hierarchical IVF + binary-partitioned `-1` subspace**
- Top-level split on "has previous tx" (dim 5/6 sentinel). Two independent IVF indices.
- Reduces inter-cluster distance noise, raises recall at same nprobe.

### C. **HNSW (pgvector / hnswlib / custom)**
- O(log N) queries, but each visit is a random pointer chase → cache-hostile, hard to keep p99 ≤ 1 ms under contention.
- Memory: typical HNSW ≈ 1.4× raw + graph (~M=16 → 64 bytes/node) = **>250 MB**. Doesn't fit duplicated; mmap helps but graph traversal randomizes accesses. **Rejected as primary.**

### D. **VP-Tree / KD-Tree (exact, no brute force)**
- KD-Tree degenerates at d≥10. VP-tree better but still ~10–30 % of N visited at high recall. Likely beats brute force only ~3×.
- **Useful as a verification oracle**, not a runtime engine.

### E. **Pure brute force on int8 + AVX2**
- 3 M × 14 = 42 M dim-ops. AVX2 doing 32×i8 mul-acc per cycle → ~1.3 M cycles ≈ ~0.5 ms on a 2.6 GHz core.
- Cache-bandwidth bound: 45 MB must stream from DRAM each query ≈ ~5 ms on dual-channel DDR3. **Too slow alone**, but the math says: *if we can reduce N by 30× via IVF, brute force on the residual is effectively free*.

**Decision**: ship A as v1; layer B on top once profiled; keep E as fallback for verification.

---

## 6. Runtime architecture proposal

```
┌─────────┐   ┌────────────────────────────────────────────┐
│  k6     │──▶│  LB (haproxy/nginx, RR, keepalive on)      │ :9999
└─────────┘   └────────┬───────────────────────┬───────────┘
                       │                       │
                  ┌────▼────┐             ┌────▼────┐
                  │  api-1  │             │  api-2  │
                  │  (zig/  │             │  (same) │
                  │   rust) │             │         │
                  └────┬────┘             └────┬────┘
                       └─────── mmap ──────────┘
                       /opt/index.bin (read-only, MAP_SHARED)
                          └─ header, centroids, postings, int8 vectors, labels
```

- **Single binary** per API: hand-tuned async HTTP server (no framework). Candidates: Rust (`hyper` minimal / `monoio` / raw epoll), Zig (`std.http` is rough; consider hand-rolled), C with `picohttpparser`. **Rust + tokio is the safe default**; Zig is the high-ceiling choice.
- **Pinning**: don't try CPU pinning under cgroup quota — let kernel schedule. Set `GOMAXPROCS`/Tokio worker threads = **1 per replica** (each replica is single-threaded). Two single-threaded replicas on one shared CPU minimize context switches.
- **Connection reuse**: LB↔upstream must use HTTP keep-alive + pre-warmed pool.
- **No TLS**, no logging, no metrics in the hot path.

---

## 7. Offline preprocessing architecture

Build stage (runs in Docker `RUN`, not on the test host):

1. Decompress `references.json.gz` (16 → 284 MB).
2. **Streaming JSON parser** (simdjson / sonic) → for each record emit 14 floats + label byte to a packed flat buffer (already 14 × 4 = 56 B + 1 B per record = 171 MB intermediate).
3. **Run k-means** (1024 centroids, 20 iterations, mini-batch). Use only ~200 k samples for centroid fit — full pass is wasteful.
4. **Assign every vector to its cell.**
5. **Sort by cell id**; within cell, sort by label (frauds first for early-exit heuristic).
6. **Quantize**: per-dimension affine `int8 = round((x - min_d) / (max_d - min_d) * 255) - 128`. Store per-dim `(min, scale)` in header (14 × 8 B = 112 B).
   - The `-1` sentinel is preserved as a special encoded value (e.g., `INT8_MIN`); a tiny mask byte per vector marks "is_null_history" so distance computation can skip dims 5/6 correctly (or compute them — both are valid choices to evaluate).
7. **Emit final binary** `index.bin`:
   - `header` (magic, version, n, d=14, n_cells, quant table, centroid array as float32 (1024×14 = 56 KB)).
   - `cell_offsets[n_cells+1]` (4 B each, ~4 KB).
   - `vectors_i8[N * 16]` (pad to 16 B for SIMD alignment; **48 MB**).
   - `labels[N]` (1 B; **3 MB**).
   - Total artifact: ~52 MB. Comfortable in RAM, mmap-friendly.
8. Compute `radius_q95` and an **optimal default `nprobe`** by replaying a held-out sample against brute force → write to header.

This artifact is copied into the runtime image at a fixed path. **Cold start = mmap + read header. < 50 ms.**

---

## 8. Memory layout proposals

Per replica resident set goal: **140 MB total**, of which:

| Region | Size | Notes |
|---|---|---|
| Code + runtime | ~20–40 MB | Rust binary stripped, no glibc bloat (musl static). |
| Tokio + buffers | ~10 MB | One worker, small slab. |
| **mmap index (shared)** | 52 MB *charged once*, but RSS counted per process | `MAP_SHARED \| MAP_POPULATE`. Kernel page cache shared between replicas. |
| Per-request scratch | < 4 KB | Stack-allocated; no per-request alloc. |
| HTTP keepalive conns | ~64 KB | LB-side only. |

**Critical**: with `MAP_SHARED` on the same file, the **page cache holds one copy**, but `ps`/cgroup RSS attribution may double-count. Verify via `/proc/<pid>/smaps` (`Pss` is the truth). The cgroup memory accounting uses **`memory.current`** which counts page cache once per cgroup; if both replicas share a cgroup, free; if separate cgroups, pages are charged to whichever touched them first. **Action: put both APIs in the same compose service scaled to 2, sharing a single cgroup parent if possible**, OR explicitly `MAP_SHARED` and accept that the second replica's mapping does not double the charge in practice (verify with a smoke test on the actual test host).

**Vector layout**: `[14 i8][2 pad] → 16 B`, aligned 16 B. 3 M vectors = 48 MB contiguous. Cells contiguous within this block. A single cell of N_cell vectors → N_cell × 16 B sequential read. At DDR3-1600 dual-channel (~12 GB/s effective), scanning the full 48 MB = ~4 ms — but we only scan ~1.5 % of it per query (≈45 k vectors per query with 16 of 1024 cells) → ~60 µs of memory bandwidth. Realistic.

---

## 9. Quantization strategies

- **int8 per-dim affine**: simplest, lossless enough for L2 since per-dim ranges are well-defined by `normalization.json`. Quant error ≈ 1/256 per dim → L2 error ≈ √14/256 ≈ 0.015. Below typical NN-gap of 0.02–0.05.
- **uint4 (nibble pack)**: 14 → 7 bytes/vec = 21 MB total. Halves bandwidth, doubles scan throughput. Likely 0.1–0.5 % recall loss. **Worth A/B testing.**
- **Binary (1 bit / dim)**: Hamming distance via `popcnt`. 14 bits/vec = 2 B. 6 MB total — fits in L2! But information loss is large; use only as a **first-pass filter** to shortlist a few hundred candidates for the int8 re-rank step.
- **Asymmetric quantization** (PQ): overkill for d=14. Skip.

**Best two-stage option**: 1-bit Hamming filter to pick top-K_coarse (~500 candidates) → int8 L2 to pick K=5. Total work per query: a 6 MB `popcnt` sweep (~5 µs with AVX2) + 500 int8 L2s (~5 µs). **Sub-50 µs ceiling possible.** Validate recall first.

---

## 10. Candidate reduction strategies

Layered, from coarse to fine:

1. **Sentinel partition**: split into has-history vs no-history. ~10–30 % reduction depending on dataset.
2. **IVF cells** (k-means or grid): nprobe=8–16 of 1024. ~50–100× reduction.
3. **Binary code prefilter**: keep top ~500 of ~30 k candidates. ~60× reduction.
4. **Exact int8 L2** on shortlisted candidates → top-5 via a fixed-size min-heap.

Each layer's recall is multiplicative — must measure end-to-end against brute force on a sample, **iteratively raising `nprobe`/`K_coarse` until recall@5 ≥ 99.5 %**.

---

## 11. Bucketization / indexing strategies

- **k-means IVF**: best recall/work ratio, costs offline k-means time only.
- **Grid (regular)**: trivial offline, predictable cell sizes, but uneven density on a real distribution wastes work. Worse.
- **Learned routing** (tiny MLP / decision tree → cell id): can predict 1–2 cells for 95 % of queries with high accuracy; gives <10 µs routing. Probably overkill for v1.
- **Within-cell ordering**: sort by L2 to centroid → enables a **distance-based early exit** when the current 5th-best already beats the lower-bound `(d_query_to_centroid − d_vec_to_centroid)`.

---

## 12. ANN vs exact-local-search tradeoffs

| Approach | Recall@5 | p99 (est.) | Memory | Verdict |
|---|---|---|---|---|
| Brute force int8 | 100 % | 0.5–2 ms | 45 MB | Backup |
| IVF + int8 (nprobe=16) | 99.7 % | 100–300 µs | 52 MB | **Primary** |
| IVF + 1-bit prefilter + int8 | 99.5 % | 30–80 µs | 58 MB | Stretch goal |
| HNSW (ef=64) | 99.8 % | 200–800 µs (cache-bound) | 250 MB | No |
| VP-Tree exact | 100 % | 300 µs–2 ms | 80 MB | No |

**Exactness of K=5 is not required**, only that the *fraud count* in top-5 matches with very high probability (since the decision is a 3-of-5 threshold). The effective accuracy budget is much larger than recall@5 suggests.

---

## 13. mmap analysis

- `MAP_SHARED | MAP_POPULATE` at startup pre-faults the 52 MB → no minor page faults in the hot path.
- Pages are clean → kernel can drop them under pressure, but with 350 MB cgroup and 52 MB resident, no pressure.
- **Huge pages (`MAP_HUGETLB` 2 MB)**: ~26 huge pages cover the index. Saves TLB pressure (Haswell L1 dTLB = 64 entries × 4 K = 256 KB coverage; 64 huge-page entries = 128 MB). **Significant** for random cell access. Requires container CAP_IPC_LOCK / sysctl `vm.nr_hugepages` — may not be allowed by the test harness. Try, fall back gracefully.
- Single shared mmap also means **one copy of the index in page cache regardless of replica count** — this is the *only* way the 350 MB budget works comfortably.

---

## 14. SIMD analysis

Target: AVX2 (Haswell). 256-bit vectors = 32 × i8 or 8 × f32.

**Hot kernel: L2² distance between query (i8[14]) and reference (i8[14])**:

```
vmovdqu  ymm0, [ref]      ; load 32 bytes (we use 14, mask the rest)
vpmovsxbw ymm1, xmm_query ; extend i8→i16
vpmovsxbw ymm2, xmm_ref
vpsubw   ymm3, ymm1, ymm2
vpmaddwd ymm4, ymm3, ymm3 ; squared diffs, accumulated as i32
vpaddd   ymm_acc, ymm_acc, ymm4
```

One reference vector per ~4 cycles. Process **4 references in parallel** with unrolling → ~1 cycle/ref. At 2.6 GHz: ~2.6 G refs/s theoretical. Practical: memory-bound at ~500 M refs/s. 30 k refs/query → **60 µs**.

Use `std::simd` (Rust nightly) or hand-written intrinsics via `core::arch::x86_64`. Zig has good intrinsic support natively. Avoid auto-vectorization gambling: write the kernel explicitly and benchmark with `perf stat -e cycles,instructions,cache-misses`.

**Top-5 selection**: a 5-element sorted insertion (4 compares worst case) per candidate is faster than any heap at this K. Branchless conditional moves preferred.

---

## 15. Cache-locality analysis

- L1d (32 KB / core, shared between 2 hyper-threads): fits ~2 k int8 vectors.
- L2 (256 KB): fits ~16 k int8 vectors.
- L3 (4 MB shared): fits ~260 k int8 vectors. **A cell of ~3 k vectors fits L1**; **16 cells (~48 k) fit L2.**
- Sequential reads within a cell → hardware prefetcher works perfectly.
- Random cell-to-cell jumps → ensure cells are 64 B-aligned (a cell-sized cache line). Pad small cells.
- **Query vector lives in registers** the whole time.
- The two replicas competing on one core thrash L1/L2 each other. The shared L3 acts as a buffer — design the working set to **fit in L3** (~3 MB scan/query). 16 cells × 3 k vec × 16 B = 768 KB — comfortable.

---

## 16. Parser / runtime analysis

JSON parsing is **non-trivially expensive** for a < 1 ms budget. A naive `serde_json::from_slice` typically costs 30–80 µs. Options:

- **simdjson** (C++ bindings or `simd-json` Rust crate): ~5–10 µs for a 400 B payload. Good.
- **Hand-rolled scanner**: payload shape is fully known and fixed-order in practice. A tape-style parser that knows exact field offsets / a state machine over byte stream → **2–4 µs**. Worth doing for top-tier numbers.
- **No allocation**: parse directly into 14 stack floats. Strings (`merchant.id`, `known_merchants`) only need membership check → compare bytes inline without allocating.
- **Date parsing**: `requested_at` only needs hour-of-day and day-of-week. Skip full ISO parse — index the bytes directly (`s[11..13]` is hour). Day-of-week from `YYYY-MM-DD` via Zeller's congruence in 4 ops.
- **`known_merchants` lookup**: `merchant.id` membership in an array of strings. Linear scan over a few strings (≤ 20) is fine; SIMD `memcmp` per pair.

**Goal: parse + normalize in < 5 µs.**

HTTP overhead:
- Avoid frameworks. `axum`/`actix` add 10–30 µs/request. Use raw `hyper` low-level or `monoio` with manual write.
- Pre-allocate response buffer (`{"approved":true,"fraud_score":0.X}` — only 3 distinct strings depending on score; can be a lookup table of 6 string responses).
- LB: **HAProxy** in TCP mode (no HTTP parsing) is the fastest. `nginx` round-robin also fine. Pre-warm connection pool.

---

## 17. Docker topology implications

- 2 services + 1 LB = 3 containers. Each container = process namespace + cgroup. Network namespace adds ~30 µs to each LB→API hop.
- CPU limits in compose (`cpus: "0.45"` × 2 + `"0.1"` LB) must sum to ≤ 1.0. Recommended: **API1 0.45, API2 0.45, LB 0.10**. The kernel CFS bandwidth controller will throttle harshly under burst — set `--cpu-period`/`--cpu-quota` with larger period (e.g., 200 ms) to reduce throttle stalls (note: docker-compose has limited support; fall back to `cpus` and accept burstiness).
- **Memory**: `mem_limit: 130M` per API, `10M` LB, fits 350 with margin for kernel slab.
- **Network mode bridge** is mandatory; cannot use `host`. Each hop adds ~20–50 µs. Total request path: client→LB (50 µs) + LB→API (50 µs) + processing (~200 µs) + back (~50 µs) ≈ **350 µs floor** before any logic.
- Place LB and APIs on the **same custom bridge network** to avoid extra NAT.

---

## 18. Resource allocation strategy

Initial split (revise after profiling):

| Service | CPU | RAM | Role |
|---|---|---|---|
| `lb` (haproxy:alpine) | 0.10 | 10 MB | TCP/HTTP RR |
| `api-1` | 0.45 | 160 MB | Inference |
| `api-2` | 0.45 | 160 MB | Inference |

Why 0.45 vs 0.5: leaves headroom for kernel + LB bursts under the 1.0 cap, avoiding CFS throttling during the 900 RPS ramp.

---

## 19. HTTP / runtime stack recommendations

**Primary recommendation: Rust + custom epoll/io_uring loop**, or `monoio` (Tokio-compatible, io_uring), or `glommio`.

- Single OS thread per replica.
- `io_uring` cuts syscall overhead vs epoll. Mac Mini Late 2014 runs Ubuntu 24.04 → kernel 6.x → io_uring available.
- Static musl binary, no allocator in the hot path (`bumpalo` for per-request if needed).
- Pre-built `hyper` is acceptable for v1; replace with hand-rolled when measuring framework overhead > 20 µs.

**Alternative: Zig.** Great for low-level control, smaller binary, but std HTTP is nascent — would write the parser/server. Higher ceiling, higher risk.

**Skip**: Go (GC pauses bad for p99), Node/Python (interpreter overhead), Java/.NET (JIT warmup, GC).

**LB**: HAProxy with `mode http`, `option http-keep-alive`, `balance roundrobin`, `maxconn 4096`. Disable logging, healthchecks at 5 s.

---

## 20. Risk analysis

| Risk | Probability | Mitigation |
|---|---|---|
| Recall drop crosses 15 % failure cutoff | Med | Validate against `test-data.json` brute force every build (CI). Conservative `nprobe`. |
| Memory cgroup OOM (mmap accounting surprise) | Med | Test on actual constraints early; use `Pss` not `Rss`. Switch to anonymous shared memory + `posix_shm` if needed. |
| CFS throttling spikes p99 | High | Single-threaded replicas, larger CPU periods, avoid background work. |
| Cold start > healthcheck timeout | Low | `MAP_POPULATE` faults pages at startup; `/ready` after that completes. 20 retries × 3 s = 60 s margin. |
| Image not linux/amd64 | Low | Pin `platform: linux/amd64` in compose; build with `--platform`. |
| Hugepages unavailable | Med | Detect at startup, fall back to 4 K pages, log nothing (quiet). |
| `known_merchants` field has 100s of entries (edge case) | Low | Cap parse at 64; use bitmap if MERC IDs are small ints. |

---

## 21. Benchmarking strategy

- **Microbench harness**: criterion.rs (Rust) for distance kernel, parser, end-to-end query (in-process, no HTTP).
- **Per-component budget table**:
  - JSON parse: 5 µs
  - Normalize+quantize query: 1 µs
  - Cell routing (compute dist to 1024 centroids in f32 SIMD): 5 µs
  - Cell scan (16 cells × ~3 k vecs × i8 L2): 50 µs
  - Top-5 + label vote: 1 µs
  - Response serialize + write: 3 µs
  - HTTP stack (in-process): 10 µs
  - LB hop + bridge network: 100 µs round-trip
  - **Total budget: ~175 µs end-to-end; p99 target ≤ 1 ms** leaves >5× safety.
- **Load harness**: `k6 run test/smoke.js` for correctness, `test.js` for full ramp on the actual test host (Mac Mini class) or a similar VM.
- **Recall harness**: replay 50 k random reference vectors as queries, compare top-5 vs brute force, compute recall@5 and *decision-agreement rate*.

---

## 22. Profiling strategy

- **`perf stat`** for cycles, IPC, cache-misses, branch-misses per query (in-process loop).
- **`perf record -F 999`** + flamegraph to find hotspots.
- **`heaptrack`** / `bytehound` to verify zero allocation in hot path.
- **`cachegrind`** for offline cache modeling.
- **`bpftrace`** for syscall counts under load (target: ~4 syscalls/req: `read`, `write`, possibly `epoll_wait`, `recvmsg`).
- **`/proc/<pid>/smaps`** to verify mmap accounting matches assumptions.
- **k6 summary JSON**: persist `results.json` per build; track p50/p99/error rate trends across commits.

---

## 23. Autotuning / specialized engine generator — feasibility

**Verdict: high upside, high risk of overengineering. Build a *minimal* version, not a real DSL.**

### What it should be
A small Python/Rust script (`tools/autotune/`) that:

1. Sweeps `(n_cells, nprobe, quant_bits, prefilter_on/off)` on the actual reference set.
2. For each config: builds the artifact, runs the recall harness + a microbench loop, records `(recall, mean_us, p99_us, mem_bytes)`.
3. Picks Pareto-optimal config(s) and writes the chosen one to `index.bin`.
4. Optionally: emits *generated Rust code* with hard-coded constants (e.g., `const N_CELLS: usize = 768;`) so the compiler can fold loops, eliminate dead branches, inline aggressively.

### What it should *not* be (yet)
- A full kernel-generating JIT.
- A C-codegen + autovec exploration framework.
- A SAT-solver-based memory-layout optimizer.

These are tempting; they are also weeks of work for **marginal** gains over a hand-tuned IVF+int8.

### When to introduce
- **v1**: hand-tuned, fixed constants.
- **v2**: parameter sweep (the minimal autotuner above).
- **v3 (only if competitive but not winning)**: code generation per-cell-size, per-quant-scheme.

### Real value
Even the v2 sweep typically finds non-obvious configurations (e.g., 768 cells > 1024 cells under our specific cache topology). The codegen step adds maybe 5–15 % to throughput by killing dynamic dispatch. Above that, diminishing returns vs the engineering cost.

---

## 24. Evolution roadmap

### Phase 0 — Skeleton (day 1)
- Repo, `docker-compose.yml`, `haproxy.cfg`, two stub Rust APIs returning `{approved:true, fraud_score:0}`.
- k6 smoke test passes. Establishes the **HTTP floor** measurement.

### Phase 1 — Correctness with brute force (day 2–3)
- Decompress + parse `references.json.gz` at build time into `vectors.f32`.
- Implement exact brute-force L2 over float32, 14 dims, single-threaded.
- Validate against `test-data.json` expected answers — must match 100 %.
- Measure p99 (expected: 3–10 ms). This is the baseline.

### Phase 2 — IVF + int8 (day 4–6)
- Offline k-means, quantization, packed binary artifact.
- mmap loader, SIMD L2 kernel.
- Recall harness, tune `nprobe`.
- Target: p99 ≤ 500 µs, recall ≥ 99.5 %.

### Phase 3 — Parser + HTTP polish (day 7–8)
- Hand-rolled JSON, response LUT.
- Strip framework overhead.
- Target: p99 ≤ 300 µs.

### Phase 4 — Two-stage prefilter (optional, day 9–10)
- 1-bit Hamming filter → int8 re-rank.
- Hugepages if available.
- Target: p99 ≤ 150 µs, recall ≥ 99.7 %.

### Phase 5 — Autotuner + final tuning (day 11–12)
- Parameter sweep, pick winning config.
- Verify on actual Mac-Mini-class hardware via preview submissions.

### Phase 6 — Submission
- Lock branch `submission`, freeze artifact, run preview, iterate.

---

## 25. Realistic path to a top-ranking score

**Conservative target**: p99 = 5 ms, FP+FN+Err rate = 0.5 %.
- p99_score: `1000 · log10(1000/5)` = **2301**
- ε ≈ (1·15 + 3·10) / 5000 = 0.009; rate_term = `1000 · log10(1/0.009)` = **2046**
- penalty ≈ `-300 · log10(46)` = **-499**
- **final_score ≈ 3848**

**Stretch target**: p99 = 1 ms, FP+FN+Err rate = 0.1 %.
- p99_score: **3000** (saturated)
- ε ≈ 0.001 → rate_term ≈ **3000** (clamped via `ε_MIN`)
- penalty ≈ `-300 · log10(6)` = **-233**
- **final_score ≈ 5767**

Top-10 positions historically cluster within 5–10 % of the ceiling. The **5800–5900 zone is the realistic competitive target**; getting there needs all of Phase 4 and most of Phase 5.

The differentiators between top-3 and top-10 are typically:
1. JSON parser µs.
2. LB hop µs (custom TCP LB beats HAProxy by ~30 µs).
3. Hugepages enabled.
4. Final `nprobe`/quantization sweet spot picked by autotuner.
5. CPU-period tuning to avoid CFS throttle spikes.

---

## Appendix A — Hard "do nots"

- Do not use a general-purpose ANN library (faiss, hnswlib) as a runtime dependency — they are tuned for batch, not 1 ms p99 single-query in 160 MB.
- Do not load `references.json.gz` at container start — preprocess at build time.
- Do not allocate per-request.
- Do not log per-request.
- Do not use TLS, compression, or middlewares.
- Do not put fraud-detection logic in the LB (banned by rules).
- Do not memoize by transaction `id` (final test uses different payloads).
- Do not use `host` or `privileged` network/mode (banned).

## Appendix B — Open questions to validate empirically

1. Does `MAP_SHARED` between sibling containers share page cache, or is it per-cgroup? (Same compose project, same network — should share, but verify with `smaps`.)
2. Is AVX2 reliably available in the test environment? (Haswell yes; container may need no special flag.)
3. Does the Mac Mini test host actually have 4 cores exposed to the cgroup, or strictly 1? (Affects whether the kernel can schedule the two replicas on different physical cores.)
4. What is the exact `requested_at` distribution — UTC always, no timezone suffixes? (Doc says UTC; verify in `example-payloads.json`.)
5. How many distinct MCC values appear at runtime vs the 10 in `mcc_risk.json`? (Affects whether to inline the lookup.)
