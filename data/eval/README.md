# Eval set

`questions.jsonl` holds 100 questions about the FastAPI 0.165.0 documentation, one JSON object per line:

| Field | Meaning |
| --- | --- |
| `id` | `q001` to `q100` |
| `question` | Asked the way a developer would ask it, often without the section's own words |
| `answer` | A short reference answer taken from the section |
| `doc_id`, `section` | The page and heading path of the section that answers the question |
| `area` | `tutorial` (49), `advanced` (31), `how-to` (8), `deployment` (7) or `core` (5) |

## How it was built

1. One substantial section was picked from each page: 300 to 3,000 characters of prose, skipping "Tip", "Info" and "Recap" boxes. That gave 108 sections. Eight were dropped because they are mostly links or screenshots, or repeat another page.
2. A question and a reference answer were written for each section from its text. The questions were drafted with help from an LLM and checked against the section.
3. `rag-eval-lab check-evalset` confirms that every page and heading path exists in the pinned corpus.

## How relevance is judged

A retrieved chunk counts as relevant when it comes from the question's page and overlaps the section's text. The section runs from its heading to the next heading of the same or a higher level. This works the same way for every chunking strategy.

## Limits

- Each question has one relevant section. Some questions are also answered elsewhere in the docs, so the scores are a lower bound.
- Questions are tied to the docs' structure, which may favour retrieval that matches heading words.
- With 100 questions, differences of 2 to 3 points between configurations are within noise.

## License

The questions and answers are original. They refer to the FastAPI documentation, which is MIT licensed.
