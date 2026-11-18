# Changelog

All notable changes to this project are documented here.
The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and the project uses [Semantic Versioning](https://semver.org/).

## [Unreleased]

## [0.1.0] - 2026-11-19

### Added

- `fetch-corpus` downloads the FastAPI 0.165.0 docs, verifies the archive checksum and keeps the English documentation pages (#7).
- `ingest` cuts pages into chunks by heading section or into fixed-size windows with overlap, without ever splitting a code block (#7).
- `index` embeds chunks with a local ONNX model and stores them in Postgres with pgvector (HNSW) and a full-text index; `docker compose up -d` starts the database (#8).
- Vector, keyword (any-word full-text) and hybrid (reciprocal rank fusion) retrievers, and a `search` command (#9).
- An eval set of 100 questions with reference answers and source sections, and a `check-evalset` command that confirms every section exists in the corpus (#10).
- `eval`, `eval-grid` and `report` commands: Hit@1/3/5/10, MRR@10 and latency for each configuration, saved per run with per-question detail (#11).
- Optional cross-encoder re-ranking of the top 30 candidates for any retriever (`--rerank` on `search`, `eval` and `eval-grid`) (#12).
- First published results: 12 configurations scored on 100 questions (#13).

### Fixed

- The vector-size check looked up the `chunks` table through the whole search path, so a table in another schema could block indexing. It now checks only the current schema (#9).
