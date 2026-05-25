# rinha-vini — submission branch

Runtime-only files for the Rinha de Backend 2026 official test.

Source code lives on the [`main` branch](https://github.com/atomosdovini/rinha-vini/tree/main).

## Architecture

1 HAProxy LB on `:9999` round-robins to 2 Rust API replicas. Each replica mmaps a shared 187 MB index file (IVF + f32 vectors). KNN-5 via AVX2/FMA L2 kernel.

Image: `ghcr.io/atomosdovini/rinha-vini-api:latest` (linux/amd64).

## Run

```bash
docker compose up -d
curl http://localhost:9999/ready
```
