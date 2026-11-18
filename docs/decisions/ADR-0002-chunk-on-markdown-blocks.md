# ADR-0002: Chunk on markdown blocks and never split code

- Status: accepted
- Date: 2026-11-19

## Context

The FastAPI docs are code-heavy. A plain character window cuts through code examples and headings, so a retrieved chunk can hold half an example, and an answer built from it cites code that does not run. Questions in the eval set point at sections, so chunks also need to map back to exact places in the page.

## Decision

Parse each page into blocks: headings, fenced code (kept whole, even with blank lines or `#` comment lines inside) and paragraphs. Both strategies pack whole blocks:

- **heading**: one chunk per heading section. Sections that hold only a heading line are skipped.
- **fixed**: windows of at most `max_chars` (default 1200) with up to `overlap` characters (default 200) repeated from the previous chunk. Only prose blocks longer than the window are split, at whitespace.

Every chunk stores `start` and `end` offsets, so `page[start:end]` is exactly the chunk text.

## Options considered

- Plain character or token windows: simplest, but they split code blocks.
- A generic recursive text splitter from an LLM framework: splits on separators, but it does not know that a fenced block must stay whole.
- A small markdown block parser (chosen): about 100 lines, fully tested, and it makes the code-block guarantee explicit.

## Consequences

- No chunk in the real corpus holds half a code block (checked on all 157 pages).
- A long code block can make a fixed-size chunk larger than `max_chars`. On FastAPI 0.165.0 that happens for 6 of 843 chunks, with the largest at 3,600 characters.
- Heading sections vary a lot in size (median 519 characters, largest 5,605), which is part of what the experiments measure.
- Setext headings (underlined with `===`) and raw HTML blocks are treated as plain text. The FastAPI docs use neither for structure.
