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
docker compose up -d                         # Postgres 17 with pgvector on port 5433
uv run rag-eval-lab index --strategy heading # embeds chunks locally, no API key needed
uv run rag-eval-lab index --strategy fixed
make check                                   # lint, types, unit and integration tests
```

`fetch-corpus` keeps the 157 English documentation pages and leaves out the changelog and internal pages. `ingest` writes one JSONL file per strategy to `data/chunks/`:

| Strategy | Chunks | Median size | Largest |
| --- | --- | --- | --- |
| heading | 1,176 | 519 chars | 5,605 chars |
| fixed (1200 / 200 overlap) | 843 | 1,118 chars | 3,600 chars (one long code block) |

No chunk splits a code block. See [ADR-0002](docs/decisions/ADR-0002-chunk-on-markdown-blocks.md) for how chunking works.

`index` embeds each chunk with [`BAAI/bge-small-en-v1.5`](https://huggingface.co/BAAI/bge-small-en-v1.5) (384 dimensions, run locally through ONNX) and stores it in one Postgres table with an HNSW vector index and a generated full-text column. Re-running `index` replaces that strategy's rows in one transaction, so it never duplicates data. On a laptop CPU it takes about two minutes per strategy.

### Try a question

```console
$ uv run rag-eval-lab search "how do I make a query parameter optional" --retriever vector -k 3
1. [0.840] tutorial/query-params.md :: Query Parameters > Optional parameters
2. [0.817] tutorial/query-params.md :: Query Parameters > Required query parameters
3. [0.794] tutorial/query-params-str-validations.md :: ... > Required parameters and `None`

$ uv run rag-eval-lab search "how do I make a query parameter optional" --retriever keyword -k 3
1. [0.744] tutorial/query-params.md :: Query Parameters > Required query parameters
2. [0.730] tutorial/query-param-models.md :: Query Parameter Models > Optional Query Parameter Models
3. [0.714] tutorial/query-params-str-validations.md :: ... > Alternative (old): `Query` as the default value
```

On this question vector search puts the right section first and keyword search does not, and hybrid search (`--retriever hybrid`, the default) follows the keyword order here. One question proves nothing, which is why every configuration is scored on the full eval set below.

## Eval set

[`data/eval/questions.jsonl`](data/eval/questions.jsonl) has 100 questions about the docs (49 tutorial, 31 advanced, 8 how-to, 7 deployment, 5 core). Each has a reference answer and the page and heading path of the section that answers it. A retrieved chunk counts as relevant when it overlaps that section, which works the same way for every chunking strategy. [data/eval/README.md](data/eval/README.md) explains how the set was built and its limits. `uv run rag-eval-lab check-evalset` confirms every section still exists in the pinned corpus. Hybrid search fuses the two rankings with reciprocal rank fusion ([ADR-0003](docs/decisions/ADR-0003-reciprocal-rank-fusion.md)).

## Design decisions

Recorded in [docs/decisions](docs/decisions).

## License

MIT
