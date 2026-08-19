"""Extract text from textbooks (PDF or DjVu), one record per page.

DjVu extraction shells out to the djvulibre CLI tools (djvused, djvutxt)
since there's no maintained pure-Python DjVu reader — pymupdf does not
support the format. Install djvulibre (e.g. `pacman -S djvulibre` /
`apt install djvulibre-bin`) for .djvu/.djv files to work.
"""

from __future__ import annotations

import json
import shutil
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path

import pymupdf as fitz

DJVU_EXTENSIONS = {".djvu", ".djv"}
PDF_EXTENSIONS = {".pdf"}


@dataclass
class Page:
    source: str
    page_number: int  # 1-indexed
    text: str


def extract_pdf(path: Path) -> list[Page]:
    pages = []
    with fitz.open(path) as doc:
        for i, page in enumerate(doc):
            text = page.get_text().strip()
            if text:
                pages.append(Page(source=path.name, page_number=i + 1, text=text))
    return pages


def _require_djvulibre() -> None:
    missing = [t for t in ("djvused", "djvutxt") if shutil.which(t) is None]
    if missing:
        raise RuntimeError(
            f"djvulibre tools not found ({', '.join(missing)}). "
            "Install djvulibre to extract .djvu/.djv files."
        )


def _djvu_page_count(path: Path) -> int:
    result = subprocess.run(
        ["djvused", "-e", "n", str(path)],
        capture_output=True,
        text=True,
        check=True,
    )
    return int(result.stdout.strip())


def extract_djvu(path: Path) -> list[Page]:
    _require_djvulibre()
    pages = []
    for i in range(1, _djvu_page_count(path) + 1):
        result = subprocess.run(
            ["djvutxt", f"-page={i}", str(path)],
            capture_output=True,
            text=True,
            check=True,
        )
        text = result.stdout.strip()
        if text:
            pages.append(Page(source=path.name, page_number=i, text=text))
    return pages


def extract_book(path: Path) -> list[Page]:
    suffix = path.suffix.lower()
    if suffix in PDF_EXTENSIONS:
        return extract_pdf(path)
    if suffix in DJVU_EXTENSIONS:
        return extract_djvu(path)
    raise ValueError(f"Unsupported book format: {path.name}")


def extract_all(raw_dir: Path, out_path: Path) -> int:
    extensions = PDF_EXTENSIONS | DJVU_EXTENSIONS
    book_paths = sorted(p for p in raw_dir.iterdir() if p.suffix.lower() in extensions)
    if not book_paths:
        raise FileNotFoundError(f"No PDF or DjVu books found in {raw_dir}")

    count = 0
    with out_path.open("w", encoding="utf-8") as f:
        for book_path in book_paths:
            for page in extract_book(book_path):
                f.write(json.dumps(asdict(page)) + "\n")
                count += 1
    return count


if __name__ == "__main__":
    import typer

    def main(
        raw_dir: Path = Path("data/raw_books"),
        out_path: Path = Path("data/processed/pages.jsonl"),
    ):
        n = extract_all(raw_dir, out_path)
        print(f"Extracted {n} pages -> {out_path}")

    typer.run(main)
