import sys
from pathlib import Path

from utils import extract_from_url, extract_from_pdf

OUTPUT_DIR = Path("./raw")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def is_url(source):
    return source.startswith("http://") or source.startswith("https://")


def is_pdf(source):
    return source.lower().endswith(".pdf")


def main():
    if len(sys.argv) < 2:
        print("Usage: uv run python extract.py <url or filepath>")
        sys.exit(1)

    source = sys.argv[1]

    if is_url(source):
        extract_from_url(source, OUTPUT_DIR)
    elif is_pdf(source):
        extract_from_pdf(source, OUTPUT_DIR)
    else:
        print("Unsupported source type. Please provide a URL or PDF file path.")
        sys.exit(1)


if __name__ == "__main__":
    main()
