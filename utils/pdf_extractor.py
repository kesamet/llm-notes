"""PDF extractor that gets text, and saves the result as a clean Markdown file."""

import os
import re
from pathlib import Path

import pymupdf


def extract_title(doc, filepath):
    metadata_title = doc.metadata.get("title", "").strip()
    if metadata_title:
        return metadata_title

    first_page = doc[0].get_text("text")
    first_line = first_page.strip().split("\n")[0].strip() if first_page.strip() else ""
    if first_line:
        return first_line

    return os.path.splitext(os.path.basename(filepath))[0]


def sanitize_filename(title):
    """Convert a title string into a safe, lowercase, hyphenated filename (max 120 chars).

    Args:
    title: The raw title string.

    Returns:
    A sanitized filename without an extension.
    """
    name = re.sub(r"[^\w\s-]", "", title)
    name = re.sub(r"[\s]+", "-", name).strip("-").lower()
    return name[:120] if name else "document"


def extract_from_pdf(filepath, output_dir):
    doc = pymupdf.open(filepath)

    title = extract_title(doc, filepath)

    pages = []
    for page in doc:
        text = page.get_text("text")
        if text.strip():
            pages.append(text.strip())

    markdown = "\n\n---\n\n".join(pages)

    header = f"# {title}\n\n> Source: {os.path.abspath(filepath)}\n\n---\n\n"
    full_md = header + markdown

    filename = sanitize_filename(Path(filepath).stem) + ".md"
    outpath = output_dir / filename
    with open(outpath, "w", encoding="utf-8") as f:
        f.write(full_md)

    print(f"Saved: {outpath}")
    return outpath
