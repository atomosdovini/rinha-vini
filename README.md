# rinha-vini

Submission for **Rinha de Backend 2026** — fraud detection via vector search.

## TL;DR

A 1-CPU / 350-MB Docker stack (1 HAProxy LB + 2 Rust API replicas, round-robin on `:9999`) that answers `POST /fraud-score` in **≤ 1 ms p99** with **0 % detection error** on a 3 M-vector reference dataset.

Architecture: **IVF (1024 k-means cells) + raw f32 vectors + AVX2/FMA L2 kernel**, mmap-shared between replicas so the 187 MB index is counted once.

See [`PLAN.md`](./PLAN.md) for the full design and [`STATUS.md`](./STATUS.md) for current measured numbers.

## Routes

- `GET /ready` — `200 ok` once the index is loaded and pre-faulted.
- `POST /fraud-score` — payload as specified in the official [API.md](https://github.com/zanfranceschi/rinha-de-backend-2026/blob/main/docs/en/API.md). Returns `{"approved": bool, "fraud_score": f}`. Never returns 5xx — falls back to safe-approve on any internal failure.

## Build & run

```bash
docker compose build       # ~50 s: fetches dataset, runs k-means, builds API
docker compose up -d
curl http://127.0.0.1:9999/ready
```

The `submission` branch contains a stripped-down `docker-compose.yml` that pulls a pre-built image from `ghcr.io/atomosdovini/rinha-vini-api:latest` (no source, no build step).

## Stack

| Layer | Choice |
|---|---|
| Language | Rust (stable, `target-cpu=haswell`) |
| HTTP | hand-rolled HTTP/1.1 on tokio current_thread (no framework) |
| JSON | hand-rolled streaming parser |
| ANN | IVF k-means, 1024 cells, default nprobe = 24 |
| Distance | exact L2² in f32 via AVX2 + FMA |
| Storage | mmap-shared, pre-faulted, 187 MB on disk |
| Load balancer | HAProxy 2.9 alpine, round-robin, keep-alive |

## License

[MIT](./LICENSE).
