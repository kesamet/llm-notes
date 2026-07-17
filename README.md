# llm-notes

A design pattern and reference scaffold for building compounding, LLM-maintained knowledge bases in plain markdown.

This repository provides a reusable implementation of the **LLM Wiki** pattern. Instead of retrieving raw documents at query time (RAG), a large language agent incrementally compiles and maintains an interlinked wiki that sits between the user and the source material. Knowledge is preserved, cross-referenced, and updated as new sources arrive, so the base compounds rather than being re-derived on every question. This README describes the architecture, core workflows, and conventions; adapt or subset it to fit a specific domain or agent.

## Motivation

Retrieval-augmented generation (RAG) is the dominant document interaction pattern: upload files, retrieve relevant chunks at query time, and generate an answer. It is flexible, but the model rediscovers relationships from scratch on every question. There is no persistent, evolving representation of what has been learned.

The LLM Wiki pattern addresses this by having the agent maintain a persistent, structured artifact: a directory of interlinked markdown files. When a new source is added, the agent extracts salient information, updates existing pages, records contradictions, and weaves the new material into the running synthesis. Cross-references, open questions, and summary judgments persist across sessions, so the knowledge base becomes richer with each source.

The human role is curatorial: choose sources, direct emphasis, ask questions, and interpret meaning. The agent handles the mechanics of summarization, cross-referencing, filing, and maintenance.

## Architecture

The system is organized into three layers:

1. **Raw sources** — immutable source documents (articles, papers, images, data files, transcripts). The agent reads from these but never modifies them. They remain the source of truth.
2. **The wiki** — agent-generated markdown files, including source summaries, entity pages, concept pages, comparison tables, and syntheses. The agent creates, updates, and cross-references these pages; the user reads them.
3. **The schema** — a project-specific instruction file such as `CLAUDE.md` or `AGENTS.md` that defines directory layout, page conventions, and workflows for ingestion, querying, and maintenance. It is treated as code and co-evolved with the wiki.

```text
raw/ # immutable source documents (articles, papers, images, data)
assets/ # downloaded images referenced by sources
wiki/ # agent-owned markdown — summaries, entities, concepts, syntheses
index.md # content-oriented catalog of every page
log.md # chronological, append-only record of activity
CLAUDE.md # schema: structure, conventions, workflows (the agent's instructions)
```

## Workflows

### Ingest

Add a source to `raw/` and instruct the agent to process it. A standard ingestion performs the following:

- Read and summarize the source.
- Discuss key takeaways with the user.
- Write or update a source summary page.
- Update `index.md` and relevant entity, concept, and synthesis pages.
- Append a dated entry to `log.md`.

A single source often touches 10–15 pages. Ingest sources individually when quality and supervision matter; batch ingestion is acceptable when throughput is preferred.

### Query

Ask questions against the wiki. The agent locates relevant pages, reads them, and synthesizes an answer with citations. Depending on the question, answers may take the form of markdown pages, comparison tables, slide decks (Marp), charts (matplotlib), or canvases.

At moderate scale (roughly 100 sources and 400K words), this approach works well without embedding-based RAG because the agent maintains an index and concise summaries. Useful answers should be filed back into the wiki so that exploratory work compounds instead of being lost to chat history.

### Lint

Periodically run a health check on the wiki. Typical concerns include:

- Contradictions between pages
- Stale claims superseded by newer sources
- Orphan pages with no inbound links
- Important concepts that lack dedicated pages
- Missing cross-references
- Gaps that could be closed with a targeted web search

The agent can also propose new questions and candidate sources to investigate.

## Index and log

### `index.md`

A content-oriented catalog of every page, with links, one-line summaries, and optional metadata, organized by category. The agent updates `index.md` on every ingest and consults it first when answering queries.

### `log.md`

A chronological, append-only record of ingests, queries, and lint passes. Entries use a consistent prefix such as:

```markdown
## [2026-04-02] ingest | Article Title
```

This makes the log parseable with standard Unix tools, for example:

```bash
grep "^## \[" log.md | tail -5
```

## Tooling

The following tools and integrations are commonly useful:

- **Obsidian** — primary reader and browsing environment; its graph view shows hubs, orphans, and clusters.
- **Obsidian Web Clipper** — converts web articles to markdown for quick ingestion.
- **Marp** — markdown-based slide decks generated from wiki content.
- **Dataview** — Obsidian plugin for querying YAML frontmatter and rendering dynamic tables.
- **Local search** — at small scale the index file is sufficient; at larger scale, a local search engine such as [qmd](https://github.com/tobi/qmd) (hybrid BM25/vector + LLM reranking) or a purpose-built script can help.

## Image handling

To ensure images remain accessible and viewable by the agent, download them locally. In Obsidian: set **Settings → Files and links → Attachment folder path** to `raw/assets/`, then bind **Settings → Hotkeys → Download attachments for current file** (for example, Ctrl+Shift+D). After clipping a web article, run the hotkey so all images are stored on disk.

## Why this works

The primary cost of maintaining a knowledge base is not reading or thinking but bookkeeping: updating cross-references, keeping summaries current, noting contradictions, and preserving consistency across many pages. Humans often abandon wikis because this overhead grows faster than the value produced. LLMs do not tire, can update many files in a single pass, and reduce the marginal cost of maintenance toward zero. The user supplies direction and judgment; the agent supplies persistence and organization.

## Getting started

1. **Download a source as markdown:**

```bash
uv run python extract.py <url or filepath>
```

2. **Ingest it with the agent**, for example in Claude Code:

```text
Create a wiki page with @<markdown.md> using the prompt: @prompts/raw-to-wiki.md
```

3. **Iterate.** Query the wiki, run lint passes, and file useful answers back into it. Co-evolve the schema (`CLAUDE.md`) as conventions stabilize.

## Scope and limitations

This pattern is intentionally abstract and the implementation modular. Directory structure, schema conventions, page formats, and tooling should be adapted to the domain, the sources, and the LLM in use. The README's purpose is to communicate the pattern; the agent and user instantiate it.

**Summary:** source documents are collected in `raw/`, compiled by an LLM into an interlinked markdown wiki in `wiki/`, and then operated on through CLI-based workflows for querying, synthesis, and incremental maintenance, typically viewed in Obsidian. The user curates and directs; the agent writes and maintains the wiki.

