"""Chunk extracted pages into overlapping passages for embedding.

Chunks are built per-source-document by concatenating pages in order and
sliding a word-count window across the text, so a chunk can span a page
boundary without losing context. Each chunk records the page range it
was drawn from for citation purposes.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

from rag.ingest import Page

CHUNK_WORDS = 220
OVERLAP_WORDS = 50


@dataclass
class Chunk:
    chunk_id: str
    source: str
    page_start: int
    page_end: int
    text: str


def load_pages(path: Path) -> list[Page]:
    pages = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            pages.append(Page(**json.loads(line)))
    return pages


def chunk_source(source: str, pages: list[Page]) -> list[Chunk]:
    # Flatten to a single list of (word, page_number) so a sliding window
    # can cross page boundaries while still tracking provenance.
    words_with_page: list[tuple[str, int]] = []
    for page in pages:
        for word in page.text.split():
            words_with_page.append((word, page.page_number))

    chunks = []
    step = CHUNK_WORDS - OVERLAP_WORDS
    idx = 0
    chunk_num = 0
    n = len(words_with_page)
    while idx < n:
        window = words_with_page[idx : idx + CHUNK_WORDS]
        if not window:
            break
        text = " ".join(w for w, _ in window)
        page_start = window[0][1]
        page_end = window[-1][1]
        chunks.append(
            Chunk(
                chunk_id=f"{source}::{chunk_num}",
                source=source,
                page_start=page_start,
                page_end=page_end,
                text=text,
            )
        )
        chunk_num += 1
        idx += step
        if idx + CHUNK_WORDS >= n and idx < n:
            # avoid a tiny dangling final chunk; let the loop finish naturally
            pass
    return chunks


def chunk_all(pages_path: Path, out_path: Path) -> int:
    pages = load_pages(pages_path)
    by_source: dict[str, list[Page]] = {}
    for p in pages:
        by_source.setdefault(p.source, []).append(p)

    count = 0
    with out_path.open("w", encoding="utf-8") as f:
        for source, source_pages in by_source.items():
            source_pages.sort(key=lambda p: p.page_number)
            for c in chunk_source(source, source_pages):
                f.write(json.dumps(asdict(c)) + "\n")
                count += 1
    return count


if __name__ == "__main__":
    import typer

    def main(
        pages_path: Path = Path("data/processed/pages.jsonl"),
        out_path: Path = Path("data/processed/chunks.jsonl"),
    ):
        n = chunk_all(pages_path, out_path)
        print(f"Created {n} chunks -> {out_path}")

    typer.run(main)
