# rag-eval-lab

[![CI](https://github.com/smitsohagiya/rag-eval-lab/actions/workflows/ci.yml/badge.svg)](https://github.com/smitsohagiya/rag-eval-lab/actions/workflows/ci.yml)

Measure how chunking, retrieval and re-ranking choices change RAG answer quality, on a fixed eval set, with the numbers published in this repo.

## Why

RAG systems are often tuned by feel: change the chunk size, try a few questions, ship it. In my day job the biggest gains came from the opposite habit: a fixed set of questions with known answers, and every change measured against it. This repo is a small, open version of that workflow, built on public documentation so anyone can rerun the numbers.

## What it compares (v0.1)

| Stage | Options |
| --- | --- |
| Chunking | fixed-size windows, heading-aware sections |
| Retrieval | vector only (pgvector), keyword only (Postgres full-text), hybrid (reciprocal rank fusion) |
| Re-ranking | none, cross-encoder |

Each run reports retrieval hit rate, answer accuracy, latency and cost. The corpus is the [FastAPI documentation](https://github.com/fastapi/fastapi/tree/master/docs/en/docs) (MIT licensed).

## How it works

See [docs/architecture.md](docs/architecture.md) for the pipeline diagram.

## Status

Early. The v0.1 roadmap is tracked in the [issues](https://github.com/smitsohagiya/rag-eval-lab/issues). Results will appear here after the first full run.

## Quickstart

```bash
make setup                                   # uv installs Python 3.12 and the dependencies
uv run rag-eval-lab fetch-corpus             # FastAPI 0.165.0 docs, checksum verified
uv run rag-eval-lab ingest --strategy heading
uv run rag-eval-lab ingest --strategy fixed --max-chars 1200 --overlap 200
make check                                   # lint, types, tests
```

`fetch-corpus` keeps the 157 English documentation pages and leaves out the changelog and internal pages. `ingest` writes one JSONL file per strategy to `data/chunks/`:

| Strategy | Chunks | Median size | Largest |
| --- | --- | --- | --- |
| heading | 1,176 | 519 chars | 5,605 chars |
| fixed (1200 / 200 overlap) | 843 | 1,118 chars | 3,600 chars (one long code block) |

No chunk splits a code block. See [ADR-0002](docs/decisions/ADR-0002-chunk-on-markdown-blocks.md) for how chunking works.

## Design decisions

Recorded in [docs/decisions](docs/decisions).

## License

MIT
