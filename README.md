# llm-notes

A design pattern and reference scaffold for building compounding, LLM-maintained knowledge bases in plain markdown.

This repository provides a reusable implementation of the **LLM Wiki** pattern. Instead of retrieving raw documents at query time (RAG), a large language agent incrementally compiles and maintains an interlinked wiki that sits between the user and the source material. Knowledge is preserved, cross-referenced, and updated as new sources arrive, so the base compounds rather than being re-derived on every question. This README describes the architecture, core workflows, and conventions; adapt or subset it to fit a specific domain or agent.

## Motivation

Retrieval-augmented generation (RAG) is the dominant document interaction pattern: upload files, retrieve relevant chunks at query time, and generate an answer. It is flexible, but the model rediscovers relationships from scratch on every question. There is no persistent, evolving representation of what has been learned.

The LLM Wiki pattern addresses this by having the agent maintain a persistent, structured artifact: a directory of interlinked markdown files. When a new source is added, the agent extracts salient information, updates existing pages, records contradictions, and weaves the new material into the running synthesis. Cross-references, open questions, and summary judgments persist across sessions, so the knowledge base becomes richer with each source.

The human role is curatorial: choose sources, direct emphasis, ask questions, and interpret meaning. The agent handles the mechanics of summarization, cross-referencing, filing, and maintenance.

## Architecture

The system is organized into two layers:

1. **Raw sources** — immutable source documents (articles, papers, images, data files, transcripts). The agent reads from these but never modifies them. They remain the source of truth.
2. **The wiki** — agent-generated source summaries. The agent creates, updates, and cross-references these pages; the user reads them.

```
llm-notes/
├── README.md # Pattern documentation and motivation
├── INDEX.md # Content-oriented catalog of every page
├── TAXONOMY.md # Controlled vocabulary for tags
├── raw/ # Layer 1: Immutable source documents
│ └── *.md # Extracted articles, papers, transcripts
├── wiki/ # Layer 2: Source summaries (1:1 with raw/)
│ └── *.md # Distilled summaries with frontmatter
├── prompts/ # Operational prompts and workflows
│ └── raw-to-wiki.md # Ingest workflow
└── scripts/ # Automation and validation
│ └── extract.py # CLI tool: URL/PDF → markdown in raw/
```

## Workflows

### Ingest

Add a source to `raw/` and instruct the agent to process it. A standard ingestion performs the following:

- Read and summarize the source.
- Discuss key takeaways with the user.
- Write or update a source summary page.
- Update `INDEX.md`.

A single source often touches 10–15 pages. Ingest sources individually when quality and supervision matter; batch ingestion is acceptable when throughput is preferred.

### Query

Ask questions against the wiki. The agent locates relevant pages, reads them, and synthesizes an answer with citations. Depending on the question, answers may take the form of markdown pages, comparison tables, slide decks (Marp), charts (matplotlib), or canvases.

At moderate scale (roughly 100 sources and 400K words), this approach works well without embedding-based RAG because the agent maintains an index and concise summaries. Useful answers should be filed back into the wiki so that exploratory work compounds instead of being lost to chat history.

## Index

### `INDEX.md`

A content-oriented catalog of every page, with links, one-line summaries, and optional metadata, organized by category. The agent updates `INDEX.md` on every ingest and consults it first when answering queries.

## Tooling

The following tools and integrations are commonly useful:

- **Obsidian** — primary reader and browsing environment; its graph view shows hubs, orphans, and clusters.
- **Obsidian Web Clipper** — converts web articles to markdown for quick ingestion.
- **Marp** — markdown-based slide decks generated from wiki content.
- **Dataview** — Obsidian plugin for querying YAML frontmatter and rendering dynamic tables.
- **Local search** — at small scale the index file is sufficient; at larger scale, a local search engine such as [qmd](https://github.com/tobi/qmd) (hybrid BM25/vector + LLM reranking) or a purpose-built script can help.

## Why this works

The primary cost of maintaining a knowledge base is not reading or thinking but bookkeeping: updating cross-references, keeping summaries current, noting contradictions, and preserving consistency across many pages. Humans often abandon wikis because this overhead grows faster than the value produced. LLMs do not tire, can update many files in a single pass, and reduce the marginal cost of maintenance toward zero. The user supplies direction and judgment; the agent supplies persistence and organization.

## Getting started

1. **Download a source as markdown:**

```bash
uv run python scripts/extract.py <url or filepath>
```

2. **Ingest it with the agent**, for example in Claude Code:

```text
Create a wiki page with @<markdown.md> using the prompt: @prompts/raw-to-wiki.md
```

## Scope and limitations

This pattern is intentionally abstract and the implementation modular. Directory structure, schema conventions, page formats, and tooling should be adapted to the domain, the sources, and the LLM in use. The README's purpose is to communicate the pattern; the agent and user instantiate it.

**Summary:** source documents are collected in `raw/`, compiled by an LLM into an interlinked markdown wiki in `wiki/`, and then operated on through CLI-based workflows for querying, synthesis, and incremental maintenance, typically viewed in Obsidian. The user curates and directs; the agent writes and maintains the wiki.
