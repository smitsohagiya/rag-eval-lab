# Architecture

```mermaid
flowchart LR
    docs[FastAPI docs, markdown] --> chunk[Chunking]
    chunk --> embed[Embeddings]
    embed --> vec[(Postgres + pgvector)]
    chunk --> fts[(Postgres full-text index)]
    question[Eval question] --> retrieve[Retrieve: vector, keyword or hybrid]
    vec --> retrieve
    fts --> retrieve
    retrieve --> rerank[Optional re-rank]
    rerank --> answer[Answer with an LLM]
    answer --> score[Score against the reference answer]
    score --> report[results/ summary]
```

- **Chunking** turns each markdown page into chunks, either fixed-size windows with overlap or one chunk per heading section.
- **Indexing** stores every chunk once, with its embedding (pgvector, HNSW index) and a generated `tsvector` column (GIN index) for keyword search. The embedded text is the chunk's heading path followed by the chunk, so short chunks keep their context.
- **Retrieval** returns the top-k chunks for a question. Hybrid fuses the vector and keyword rankings with reciprocal rank fusion.
- **Re-ranking** optionally re-orders the top candidates with a cross-encoder before they reach the LLM.
- **Evaluation** runs every eval question through one configuration and records whether the right source section was retrieved, whether the answer matches the reference, the latency and the token cost.
