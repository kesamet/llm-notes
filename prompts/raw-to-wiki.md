You are converting a raw extracted article into a structured wiki page for a personal knowledge base about LLM architectures, training, and AI research.

## Input
- The raw article markdown (provided below)
- Existing wiki pages in the knowledge base (provided below for cross-referencing)

## Output format

### YAML Frontmatter
Every summary page must begin with YAML frontmatter:

```yaml
---
title: "<Article Title>"
type: summary
source: "raw/<filename>.md"
date_ingested: YYYY-MM-DD
tags: [domain-tag, technique-tag, ...]
concepts: [concept1, concept2, ...]
entities: [entity1, entity2, ...]
status: draft
related: [page1.md, page2.md]
---
```

**Field definitions:**
- `title`: Human-readable article title
- `type`: Always "summary" for source summaries
- `source`: Path to raw source file
- `date_ingested`: Today's date in ISO 8601 format
- `tags`: Use controlled vocabulary from `wiki/TAXONOMY.md`
- `concepts`: List of concepts covered (use tags from taxonomy)
- `entities`: List of entities mentioned (models, orgs, tools)
- `status`: Always "draft" for new pages
- `related`: List of related pages (will be filled during cross-referencing)

### Page Structure
1. **Title line**: `# <Article Title> -- Wiki`
2. **Attribution block**: `> Based on <author>'s article (<month year>)` followed by `> Source: <url>`
3. **Table of Contents**: Full linked TOC using `- [Section](#anchor)` with nesting
4. **Sections separated by `---` horizontal rules**
5. **Evolution section**: Add before References section (see below)
6. **References section**: List all papers, repos, and links mentioned

## Content guidelines

- **Distill, don't copy.** Rewrite the article's insights into concise, reference-friendly prose. Strip promotional content, subscription CTAs, and image-only references.
- **Use comparison tables liberally.** Whenever the article contrasts two or more approaches, models, or techniques, present them in a markdown table with clear column headers.
- **Preserve quantitative detail.** Keep specific numbers (parameter counts, benchmark scores, ratios, dates) -- these are what make a wiki entry useful for quick lookup.
- **Document evidence.** For quantitative claims, note the source (table, figure, section) and benchmark used.
- **Add a "Key Takeaways" section** with 5-8 numbered, opinionated summaries of what matters most.
- **Add an "Evolution" section** before References:

```markdown
## Evolution

- **YYYY-MM-DD**: Initial extraction from <source>. Documented <key patterns/findings>.
```

## Cross-referencing the existing knowledge base

Review the existing wiki pages. Where the new article covers a topic already present:

### Don't Repeat
Summarize briefly and link to the existing page using wikilinks:
```markdown
For detailed coverage of MLA, see [[multi-head-latent-attention|Multi-Head Latent Attention]].
```

### Add New Information
Update existing pages with:
- Newer models or benchmarks
- Different perspectives or additional detail
- Evidence that confirms or contradicts existing claims

### Update Related Pages
For each related concept/entity/comparison/synthesis page:
1. Add new evidence to the "Evidence Trail" table
2. Add dated entry to the "Evolution" section
3. Update frontmatter `related` field if needed

### Create New Pages
If the article introduces a major new concept or entity not covered:
- Create concept page in `wiki/concepts/`
- Create entity page in `wiki/entities/`
- Create comparison page in `wiki/comparisons/` if comparing multiple approaches
- Create synthesis page in `wiki/syntheses/` if integrating multiple sources

### Use Wikilinks
Replace prose cross-references with Obsidian wikilinks:
```markdown
# Instead of:
For details, see [Page Title](page-filename.md)

# Use:
For details, see [[page-filename|Page Title]]
```

## Evidence Requirements

For quantitative claims, document:
- **Source citation**: which raw file, which table/figure/section
- **Benchmark name**: which benchmark was used
- **Metric**: what was measured (accuracy, latency, throughput, etc.)
- **Comparison baseline**: what was it compared against
- **Confidence level**:
- `high`: empirical results, multiple benchmarks, peer-reviewed
- `medium`: empirical results, single benchmark, credible source
- `low`: anecdotal, theoretical, or unverified claims

## Style rules
- No emojis
- ATX-style headings (`#`, `##`, `###`)
- Pipe tables for comparisons (with `|---|` separator rows)
- Bold key terms on first meaningful use within a section
- Keep paragraphs short (3-5 sentences max)
- Use `code formatting` for model names, hyperparameter values, and code references

## Post-Ingest Checklist

After creating the summary page:

1. ✅ Update `index.md` with new entry
2. ✅ Append entry to `log.md`: `## [YYYY-MM-DD] ingest | <Title>`
3. ✅ Update related concept/entity/comparison/synthesis pages
4. ✅ Add pattern extraction to `wiki/EVOLUTION_LOG.md`
5. ✅ Run `python scripts/lint_kb.py` to verify consistency

---

### Existing wiki pages:

Read `index.md` for a condensed index of all existing wiki pages with their topics, key terms, and covered models. Refer to individual files in `summaries/` for full detail when needed.

Read `wiki/TAXONOMY.md` for the controlled vocabulary of tags.

Read `wiki/EVOLUTION_LOG.md` for existing pattern extractions.

### Raw article to convert:

<paste contents of the raw file>
