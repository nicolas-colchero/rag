"""Convert logged relevance feedback into fine-tuning pairs.

Reads the feedback log written by `rag.retrieve`/`rag.repl --feedback`
(see rag.feedback) and, for every +1-rated result, looks up the chunk's
full text in the index's chunks.jsonl and emits a {"question", "passage"}
row shaped for rag.finetune.

-1-rated results are skipped: MultipleNegativesRankingLoss (rag.finetune)
only trains on positives via in-batch negatives, so there's currently no
consumer for explicit hard negatives.
"""

from __future__ import annotations

import json
from pathlib import Path

from rag.chunk import Chunk
from rag.feedback import DEFAULT_FEEDBACK_PATH, Feedback, load_feedback

DEFAULT_CHUNKS_PATH = Path("data/index/chunks.jsonl")
DEFAULT_OUT_PATH = Path("data/finetune_pairs.jsonl")


def load_chunk_texts(chunks_path: Path) -> dict[str, Chunk]:
    chunks = {}
    with chunks_path.open(encoding="utf-8") as f:
        for line in f:
            c = Chunk(**json.loads(line))
            chunks[c.chunk_id] = c
    return chunks


def load_existing_keys(out_path: Path) -> set[tuple[str, str]]:
    if not out_path.exists():
        return set()
    keys = set()
    with out_path.open(encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            keys.add((row["question"], row.get("chunk_id", "")))
    return keys


def build_pairs(
    feedback: list[Feedback], chunks: dict[str, Chunk]
) -> tuple[list[dict], int, int]:
    """Returns (pairs, n_negative_skipped, n_missing_chunk_skipped)."""
    pairs = []
    n_negative = 0
    n_missing = 0
    for entry in feedback:
        if entry.score != 1:
            n_negative += 1
            continue
        chunk = chunks.get(entry.chunk_id)
        if chunk is None:
            n_missing += 1
            continue
        pairs.append(
            {
                "question": entry.question,
                "passage": chunk.text,
                "chunk_id": chunk.chunk_id,
                "source": chunk.source,
                "page_start": chunk.page_start,
                "page_end": chunk.page_end,
            }
        )
    return pairs, n_negative, n_missing


def write_pairs(pairs: list[dict], out_path: Path, append: bool) -> int:
    """Writes new pairs, deduped against anything already at out_path.

    Returns the number of rows actually written.
    """
    existing_keys = load_existing_keys(out_path) if append else set()
    new_rows = [p for p in pairs if (p["question"], p["chunk_id"]) not in existing_keys]

    out_path.parent.mkdir(parents=True, exist_ok=True)
    mode = "a" if append and out_path.exists() else "w"
    with out_path.open(mode, encoding="utf-8") as f:
        for row in new_rows:
            f.write(json.dumps(row) + "\n")
    return len(new_rows)


if __name__ == "__main__":
    import typer

    def main(
        feedback_path: Path = DEFAULT_FEEDBACK_PATH,
        chunks_path: Path = DEFAULT_CHUNKS_PATH,
        out_path: Path = DEFAULT_OUT_PATH,
        append: bool = True,
    ):
        feedback = load_feedback(feedback_path)
        if not feedback:
            raise SystemExit(f"No feedback found in {feedback_path}")

        chunks = load_chunk_texts(chunks_path)
        pairs, n_negative, n_missing = build_pairs(feedback, chunks)

        n_written = write_pairs(pairs, out_path, append=append)
        n_duplicate = len(pairs) - n_written

        print(f"{len(feedback)} feedback entries -> {len(pairs)} positive pairs")
        if n_negative:
            print(f"  skipped {n_negative} negative (-1) ratings (not usable by finetune's loss)")
        if n_missing:
            print(f"  skipped {n_missing} whose chunk_id was not found in {chunks_path}")
        if n_duplicate:
            print(f"  skipped {n_duplicate} already present in {out_path}")
        print(f"Wrote {n_written} new pairs -> {out_path}")

    typer.run(main)
