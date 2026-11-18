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
make setup   # uv installs Python 3.12 and the dependencies
make check   # lint, types, tests
```

## Design decisions

Recorded in [docs/decisions](docs/decisions).

## License

MIT
