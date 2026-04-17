# AGENTS

This repository is a personal LLM learning wiki. Treat it as a persistent knowledge base, not a scratchpad.

## Core model

There are three layers:

1. `raw/`: immutable source material. Read from here. Do not rewrite source contents.
2. `wiki/`: LLM-maintained markdown knowledge base. This is the main artifact to update.
3. `AGENTS.md`: the schema and operating rules for maintaining the wiki.

## Primary goals

When working in this repo, optimize for accumulation:

1. Convert raw sources into durable wiki pages.
2. Update existing pages instead of duplicating the same idea in many places.
3. Maintain cross-links between pages.
4. Keep a compact content index in `wiki/index.md`.
5. Append a chronological record in `wiki/log.md`.

## Directory conventions

- `raw/inbox/`: newly added sources waiting for ingestion
- `raw/processed/`: sources already ingested
- `raw/assets/`: local images and attachments
- `wiki/00-overview/`: high-level entry pages and maps
- `wiki/10-roadmaps/`: reading plans and staged study routes
- `wiki/20-concepts/`: concept pages
- `wiki/30-papers/`: individual paper pages
- `wiki/40-comparisons/`: compare/contrast pages
- `wiki/50-questions/`: open questions, hypotheses, TODOs
- `wiki/90-meta/`: repo maintenance notes, lint reports, conventions
- `templates/`: page templates

## Page rules

Prefer short, structured markdown pages.

Each durable wiki page should usually have:

1. A title
2. A one-sentence summary near the top
3. Source links or source references
4. Internal links to related pages when relevant
5. A final section for open questions, tensions, or next reads when useful

## Ingest workflow

When the user asks to ingest a source:

1. Read the source from `raw/inbox/` or a provided URL/file.
2. Decide which existing wiki pages should be updated.
3. Create or update:
   - one source-specific page if needed
   - affected concept pages
   - affected comparison or roadmap pages
   - `wiki/index.md`
   - `wiki/log.md`
4. Move the raw file from `raw/inbox/` to `raw/processed/` only if the user wants file organization handled automatically.

## Query workflow

When the user asks a question:

1. Read `wiki/index.md` first.
2. Read the most relevant wiki pages.
3. Synthesize an answer from the wiki, not directly from memory when possible.
4. If the answer creates durable value, offer to file it back into the wiki as a new or updated page.

## Lint workflow

When asked to lint or health-check the wiki, look for:

1. orphan pages
2. stale claims
3. duplicated notes that should be merged
4. missing cross-links
5. empty sections or TODO-heavy pages
6. important concepts referenced repeatedly but lacking their own page

Write lint findings to `wiki/90-meta/`.

## Style

Prefer substance over polish.

Do not turn the wiki into a diary of chat transcripts. Convert chat output into durable notes.

Avoid redundant pages when an update to an existing page is better.

Prefer markdown links and relative paths inside the repo.
