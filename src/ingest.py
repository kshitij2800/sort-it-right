"""Brick 4: turn the downloaded sorting guides into small text chunks for the search index.

Reads data/sources.csv + the files in data/raw/ (run src/download_sources.py first)
and writes data/chunks.jsonl: one JSON object per line = one chunk with its source.

Run:
    python src/ingest.py
"""
import json

import pandas as pd
from bs4 import BeautifulSoup
from pypdf import PdfReader

SOURCES_PATH = "data/sources.csv"
RAW_DIR = "data/raw"
CHUNKS_PATH = "data/chunks.jsonl"
CHUNK_SIZE = 500      # characters per chunk (about 80 words)
CHUNK_OVERLAP = 100   # characters repeated from the end of the previous chunk

# Where the real content is in each website's HTML (found by inspecting the pages).
# The key is the start of the file name in data/raw/, e.g. "metsasairila_metal.html".
CONTENT_SELECTORS = {
    "metsasairila": ".js-hide-asiointi",   # page text without the opening-hours box
    "rinki": "article",
    "palpa": ".main-content-area",
    "infofinland": "#content",
}


def extract_text(file):
    """Return the readable text of one downloaded file (PDF or HTML)."""
    path = f"{RAW_DIR}/{file}"

    if file.endswith(".pdf"):
        pages = PdfReader(path).pages
        return "\n".join(page.extract_text() for page in pages)

    soup = BeautifulSoup(open(path, "rb").read(), "html.parser")
    # Remove parts that are never sorting information.
    for tag in soup(["script", "style", "nav", "header", "footer", "form"]):
        tag.decompose()
    site = file.split("_")[0]                         # "metsasairila_metal.html" -> "metsasairila"
    parts = soup.select(CONTENT_SELECTORS[site])
    # "\n" between pieces keeps list items on their own lines.
    return "\n".join(part.get_text("\n", strip=True) for part in parts)


def clean_text(text):
    """Squash repeated spaces inside each line and drop empty lines."""
    lines = [" ".join(line.split()) for line in text.splitlines()]
    return "\n".join(line for line in lines if line)


def chunk_text(text, size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    """Cut text into pieces of about `size` characters; neighbours share about `overlap` characters."""
    chunks = []
    start = 0
    while start < len(text):
        end = min(start + size, len(text))
        if end < len(text):
            # Don't cut a word in half: step back to the last space or line break.
            cut = max(text.rfind(" ", start, end), text.rfind("\n", start, end))
            if cut > start:
                end = cut
        chunks.append(text[start:end].strip())
        if end == len(text):
            break
        # The next chunk starts `overlap` characters earlier, moved forward to the start of a word.
        start = end - overlap
        while not text[start - 1].isspace():
            start += 1
    return chunks


if __name__ == "__main__":
    sources = pd.read_csv(SOURCES_PATH, dtype=str)
    total = 0

    with open(CHUNKS_PATH, "w", encoding="utf-8") as out:
        for source in sources.itertuples():
            text = clean_text(extract_text(source.file))
            chunks = chunk_text(text)
            for n, chunk in enumerate(chunks):
                record = {
                    "id": f"{source.file}#{n}",      # unique id, e.g. "metsasairila_metal.html#0"
                    "text": chunk,
                    "file": source.file,
                    "title": source.title,         # shown as the source under the answer
                    "url": source.url,
                }
                # ensure_ascii=False keeps "ä" readable instead of "ä".
                out.write(json.dumps(record, ensure_ascii=False) + "\n")
            print(f"{len(chunks):3} chunks  {len(text):5} chars  {source.file}")
            total += len(chunks)

    print(f"\n{total} chunks saved to {CHUNKS_PATH}")
