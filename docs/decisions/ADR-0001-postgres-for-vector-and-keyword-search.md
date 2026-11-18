# ADR-0001: Keep vector and keyword search in one Postgres database

- Status: accepted
- Date: 2026-11-19

## Context

Hybrid retrieval needs a vector index and a keyword index. Running a separate vector database and a separate search engine doubles the setup for anyone who wants to rerun the experiments.

## Decision

Use Postgres with pgvector for embeddings and Postgres full-text search (`tsvector` with a GIN index) for keywords, and fuse the two ranked lists with reciprocal rank fusion in application code.

## Options considered

- Postgres for both (chosen): one container, one connection, and one transaction covers both indexes.
- pgvector plus Elasticsearch: better keyword relevance tuning, but a second service to run and keep in sync.
- A hosted vector database: no local setup, but it needs an account and an API key, which makes the results harder to reproduce.

## Consequences

- `docker compose up` will start everything the experiments need.
- Postgres full-text ranking (`ts_rank`) is simpler than BM25, so keyword-only results may understate what a dedicated search engine could do. The results section will say so next to the numbers.
