# Changelog

All notable changes to this project are documented here.
The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and the project uses [Semantic Versioning](https://semver.org/).

## [Unreleased]

### Added

- `fetch-corpus` downloads the FastAPI 0.165.0 docs, verifies the archive checksum and keeps the English documentation pages (#7).
- `ingest` cuts pages into chunks by heading section or into fixed-size windows with overlap, without ever splitting a code block (#7).
- `index` embeds chunks with a local ONNX model and stores them in Postgres with pgvector (HNSW) and a full-text index; `docker compose up -d` starts the database (#8).
- Vector, keyword (any-word full-text) and hybrid (reciprocal rank fusion) retrievers, and a `search` command (#9).
- An eval set of 100 questions with reference answers and source sections, and a `check-evalset` command that confirms every section exists in the corpus (#10).

### Fixed

- The vector-size check looked up the `chunks` table through the whole search path, so a table in another schema could block indexing. It now checks only the current schema (#9).
