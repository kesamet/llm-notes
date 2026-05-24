# llm-notes

Tools and prompts for extracting research papers, technical articles, and PDFs into clean Markdown, and formatting them into a structured personal wiki/knowledge base.

## Features

- **Web Article Extraction**: Fetches and isolates core content of any article/blog post, strips layouts/boilerplates, and generates clean Markdown.
- **PDF Extraction**: Converts local PDFs, extraction of titles, and outputs formatted Markdown page-by-page.
- **Wiki Transformation**: Systematic guidelines and prompts to transform the raw text into structured wiki reference pages with comparison tables, quantitative details, robust cross-referencing, and action items.

## Setup

Make sure you have [uv](https://github.com/astral-sh/uv) installed, then run the commands directly with `uv run`.

## Usage

### 1. Extract Source Documents

Run the extractor directly by passing either a URL or a file path to a PDF:

```bash
# Extract from a web article
uv run python extract.py https://example.com/some-article-on-llms

# Extract from a local PDF file
uv run python extract.py path/to/paper.pdf
```

The script will automatically detect the source type and save the extracted Markdown file under the `raw/` directory.

### 2. Formulating Wiki Pages

To transform the raw extracted markdown into a polished personal wiki page, use the prompt template `prompts/raw-to-wiki.md`.

Copy the pattern from the prompt and supply:
1. **Existing context/wiki articles** (located in the `summaries/` directory) for cross-referencing and consistent terminology.
2. **The raw extracted Markdown** from the `raw/` directory.

Feed these inputs to any advanced LLM (e.g. Claude 3.5 Sonnet) to receive a well-structured wiki entry ready to index in your knowledge base.
