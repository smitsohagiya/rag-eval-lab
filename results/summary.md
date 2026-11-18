# Results

Corpus: FastAPI 0.165.0 docs. Questions: 100. Embedding model: `BAAI/bge-small-en-v1.5`.

Hit@k is the share of questions with a chunk from the answering section in the top k. MRR@10 is the mean of 1 / rank of the first such chunk (0 if it is not in the top 10).

| Chunking | Retrieval | Re-rank | Hit@1 | Hit@3 | Hit@5 | Hit@10 | MRR@10 | p50 latency |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| fixed | hybrid | yes | 72% | 96% | 97% | 99% | 0.833 | 2005 ms |
| fixed | vector | yes | 72% | 95% | 95% | 97% | 0.827 | 1904 ms |
| fixed | keyword | yes | 70% | 91% | 93% | 95% | 0.806 | 1904 ms |
| heading | vector | yes | 70% | 89% | 92% | 97% | 0.798 | 1768 ms |
| fixed | vector | no | 68% | 90% | 94% | 97% | 0.797 | 10 ms |
| heading | hybrid | yes | 68% | 88% | 90% | 96% | 0.783 | 2026 ms |
| heading | keyword | yes | 67% | 88% | 92% | 95% | 0.782 | 1866 ms |
| fixed | hybrid | no | 65% | 86% | 93% | 98% | 0.770 | 21 ms |
| heading | hybrid | no | 63% | 83% | 92% | 98% | 0.744 | 20 ms |
| heading | vector | no | 62% | 80% | 88% | 95% | 0.729 | 8 ms |
| fixed | keyword | no | 33% | 66% | 80% | 86% | 0.509 | 7 ms |
| heading | keyword | no | 30% | 63% | 67% | 84% | 0.481 | 7 ms |

## Best configuration by area: fixed-hybrid-rerank

| Area | Questions | Hit@5 |
| --- | --- | --- |
| advanced | 31 | 97% |
| core | 5 | 100% |
| deployment | 7 | 100% |
| how-to | 8 | 75% |
| tutorial | 49 | 100% |
