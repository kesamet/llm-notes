You are converting a raw extracted article into a structured wiki page for a personal knowledge base about LLM architectures, training, and AI research.

## Input
- The raw article markdown (provided below)
- Existing wiki pages in the knowledge base (provided below for cross-referencing)

## Output format
Produce a single markdown file following these conventions:

1. **Title line**: `# <Article Title> -- Wiki`
2. **Attribution block**: `> Based on <author>'s article (<month year>)` followed by `> Source: <url>`
3. **Table of Contents**: Full linked TOC using `- [Section](#anchor)` with nesting
4. **Sections separated by `---` horizontal rules**

## Content guidelines

- **Distill, don't copy.** Rewrite the article's insights into concise, reference-friendly prose. Strip promotional content, subscription CTAs, and image-only references.
- **Use comparison tables liberally.** Whenever the article contrasts two or more approaches, models, or techniques, present them in a markdown table with clear column headers.
- **Preserve quantitative detail.** Keep specific numbers (parameter counts, benchmark scores, ratios, dates) -- these are what make a wiki entry useful for quick lookup.
- **Add a "Key Takeaways" section** at the end with numbered, opinionated summaries of what matters most from the article.
- **End with a "References" section** listing all papers, repos, and links mentioned in the article as bare URLs or `name: url` pairs.

## Cross-referencing the existing knowledge base

Review the existing wiki pages provided. Where the new article covers a topic already present in the knowledge base:
- **Don't repeat** explanations already covered well elsewhere. Summarize briefly and note that the topic is covered in more detail in the other page.
- **Add new information** that the existing pages don't have (newer models, updated benchmarks, different perspectives, additional detail).
- **Use consistent terminology** with the existing pages (e.g., if the KB calls it "MLA" don't switch to "Multi-Head Latent Attention" without the abbreviation).

## Style rules
- No emojis
- ATX-style headings (`#`, `##`, `###`)
- Pipe tables for comparisons (with `|---|` separator rows)
- Bold key terms on first meaningful use within a section
- Keep paragraphs short (3-5 sentences max)
- Use `code formatting` for model names, hyperparameter values, and code references

---

### Existing wiki pages:

<paste contents of each file in summaries/>

### Raw article to convert:

<paste contents of the raw file>

