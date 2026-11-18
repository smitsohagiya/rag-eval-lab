# ADR-0003: Fuse vector and keyword results with reciprocal rank fusion

- Status: accepted
- Date: 2026-11-19

## Context

Hybrid retrieval combines two ranked lists whose scores mean different things. Vector search returns cosine similarity (about 0.6 to 0.9 on this corpus), and Postgres `ts_rank_cd` returns a normalised term-overlap score. Adding the raw scores lets one retriever drown out the other, and min-max normalisation shifts from question to question.

## Decision

Take the top 50 candidates from each retriever and fuse them with reciprocal rank fusion: each list adds `1 / (60 + rank)` to a chunk, and the fused list is sorted by the total. `k = 60` is the value from the original RRF paper (Cormack, Clarke and Büttcher, 2009).

## Options considered

- Weighted sum of normalised scores: tunable, but the weight has to be tuned per corpus, and normalisation is sensitive to outliers.
- Reciprocal rank fusion (chosen): no tuning, and it only uses positions, so the two scales never meet.
- Skip fusion and rely on a re-ranker: covered separately by #12. The re-ranker still needs a good candidate list, and fusion gives it one.

## Consequences

- No per-corpus tuning, and the fused scores are small numbers (about 0.016 to 0.033) that only order results.
- Score magnitudes are ignored: a near-perfect keyword match at rank 1 counts the same as a weak rank 1. The eval set will show whether that costs accuracy on this corpus.
- Each question runs two queries of 50 candidates each, which is cheap at this corpus size.
