"""Web article extractor that fetches a URL, isolates the main content,
strips navigation and boilerplate, and saves the result as a clean Markdown file.
"""

import re

import requests
from bs4 import BeautifulSoup
from markdownify import markdownify

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    )
}


def fetch_page(url):
    """Fetch the raw HTML content of a web page.

    Args:
    url: The URL to fetch.

    Returns:
    The HTML content as a string.

    Raises:
    requests.HTTPError: If the HTTP response status indicates an error.
    """
    resp = requests.get(url, headers=HEADERS, timeout=30)
    resp.raise_for_status()
    return resp.text


def find_article_content(soup):
    """Locate the main content element in a parsed HTML page.

    Tries three strategies in order: <article> tag, <main>/role="main",
    then falls back to the largest <div> by text length.

    Args:
    soup: A BeautifulSoup object of the full page.

    Returns:
    A BeautifulSoup Tag containing the primary content.
    """
    # Strategy 1: <article> tag
    article = soup.find("article")
    if article:
        return article

    # Strategy 2: <main> or role="main"
    main = soup.find("main") or soup.find(attrs={"role": "main"})
    if main:
        return main

    # Strategy 3: largest <div> by text length
    divs = soup.find_all("div")
    if divs:
        return max(divs, key=lambda d: len(d.get_text(strip=True)))

    return soup.body or soup


def extract_title(soup):
    """Extract the page title using og:title, <title>, or <h1> as fallbacks.

    Args:
    soup: A BeautifulSoup object of the full page.

    Returns:
    The page title string, or "untitled" if none is found.
    """
    og_title = soup.find("meta", property="og:title")
    if og_title and og_title.get("content"):
        return og_title["content"].strip()

    title_tag = soup.find("title")
    if title_tag:
        return title_tag.get_text(strip=True)

    h1 = soup.find("h1")
    if h1:
        return h1.get_text(strip=True)

    return "untitled"


def clean_content(soup_element):
    """Remove non-content elements (nav, footer, sidebars, etc.) in place.

    Args:
    soup_element: A BeautifulSoup Tag to clean. Modified in place.
    """
    for tag in soup_element.find_all(["nav", "footer", "header", "aside", "script", "style", "noscript"]):
        tag.decompose()

    attrs = {"class": re.compile(r"sidebar|menu|nav|footer|comment|share|social|subscribe|popup", re.I)}
    for tag in soup_element.find_all(attrs=attrs):
        tag.decompose()


def to_markdown(soup_element):
    """Convert an HTML element to cleaned Markdown text.

    Args:
    soup_element: A BeautifulSoup Tag to convert.

    Returns:
    A Markdown string with excess blank lines collapsed.
    """
    raw = markdownify(str(soup_element), heading_style="ATX", code_language="python", strip=["button", "input", "form"])
    # Collapse 3+ blank lines into 2
    cleaned = re.sub(r"\n{3,}", "\n\n", raw)
    return cleaned.strip()


def sanitize_filename(title):
    """Convert a title string into a safe, lowercase, hyphenated filename (max 120 chars).

    Args:
    title: The raw title string.

    Returns:
    A sanitized filename without an extension.
    """
    name = re.sub(r"[^\w\s-]", "", title)
    name = re.sub(r"[\s]+", "-", name).strip("-").lower()
    return name[:120] if name else "article"


def extract_from_url(url, output_dir):
    """Fetch a web page, extract its main content, and save it as Markdown.

    Args:
    url: The URL of the page to extract.
    output_dir: Directory to write the output file into. Created if missing.

    Returns:
    The file path of the saved Markdown file.
    """
    html = fetch_page(url)
    soup = BeautifulSoup(html, "html.parser")

    title = extract_title(soup)
    content = find_article_content(soup)
    clean_content(content)
    markdown = to_markdown(content)

    # Prepend title and source URL
    header = f"# {title}\n\n> Source: {url}\n\n---\n\n"
    full_md = header + markdown

    filename = sanitize_filename(title) + ".md"
    outpath = output_dir / filename
    with open(outpath, "w", encoding="utf-8") as f:
        f.write(full_md)

    print(f"Saved: {outpath}")
    return outpath
