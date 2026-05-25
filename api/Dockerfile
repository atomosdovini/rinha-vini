# Multi-stage build for the Rinha de Backend 2026 fraud-detection API.
#
# Stages:
#   1. preprocess+build: fetch `references.json.gz` from the official repo,
#      compile `tools/preprocess`, produce `/opt/index.bin`, then compile the
#      API binary.
#   2. runtime: minimal image with just the API binary + index.bin.
#
# The image targets linux/amd64 (Haswell baseline → AVX2 + FMA).

FROM --platform=linux/amd64 rust:1.83-slim AS builder

# System deps for fetching the dataset.
RUN apt-get update && apt-get install -y --no-install-recommends curl ca-certificates \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /work

# 1) Fetch the official 3 M-vector reference dataset. Version pinned to the
#    repo's main branch — same file every build.
ARG REFERENCES_URL=https://raw.githubusercontent.com/zanfranceschi/rinha-de-backend-2026/main/resources/references.json.gz
RUN curl -fsSL "$REFERENCES_URL" -o /work/references.json.gz \
 && ls -lh /work/references.json.gz

# 2) Build the offline preprocessor and run it once to produce /opt/index.bin.
COPY tools/preprocess /work/preprocess
RUN cd /work/preprocess && cargo build --release
RUN mkdir -p /opt \
 && /work/preprocess/target/release/preprocess /work/references.json.gz /opt/index.bin \
 && ls -lh /opt/index.bin

# 3) Build the API binary with AVX2 + FMA enabled (Haswell baseline).
COPY api /work/api
ENV RUSTFLAGS="-C target-cpu=haswell"
RUN cd /work/api && cargo build --release

# ── runtime ──────────────────────────────────────────────────────────────────
FROM --platform=linux/amd64 debian:bookworm-slim AS runtime
COPY --from=builder /work/api/target/release/rinha-api /usr/local/bin/rinha-api
COPY --from=builder /opt/index.bin /opt/index.bin
ENV LISTEN_ADDR=0.0.0.0:8080 \
    INDEX_PATH=/opt/index.bin
EXPOSE 8080
ENTRYPOINT ["/usr/local/bin/rinha-api"]
