# Knowledge Base Overview

This repo is a personal LLM knowledge base designed for cumulative learning rather than repeated ad hoc retrieval.

## Core idea

Instead of re-deriving understanding from raw papers and links every time, the repo maintains a persistent markdown wiki between raw sources and future questions.

The human curates sources and asks good questions.

The LLM maintains summaries, concept pages, comparisons, roadmaps, and cross-links.

## Layers

1. `raw/`: source of truth for imported material
2. `wiki/`: synthesized knowledge layer
3. `AGENTS.md`: schema and workflow rules

## Operating principles

1. New knowledge should be filed into durable pages.
2. Existing pages should be updated when they already cover the topic.
3. The index should stay compact and navigable.
4. The log should record notable ingests, queries, and maintenance passes.
5. A good answer in chat is often worth turning into a wiki page.

## What belongs here

Good fits:

1. paper summaries
2. concept clarifications
3. reading plans
4. side-by-side comparisons
5. open questions
6. evolving conclusions

Bad fits:

1. long raw chat transcripts
2. duplicated copies of raw papers
3. notes with no clear purpose or links into the rest of the wiki
