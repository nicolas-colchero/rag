"""Log user relevance feedback on retrieved chunks.

Feedback is scored 0 / +1 / -1:
  0  -> no signal, nothing is recorded
 +1  -> chunk answered the question (a positive pair for later fine-tuning)
 -1  -> chunk did NOT answer the question (a hard negative for later fine-tuning)

This module only appends to a log; it does not touch the model or the
index. The log is the dataset a future fine-tuning run will consume.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

DEFAULT_FEEDBACK_PATH = Path("data/feedback.jsonl")


@dataclass
class Feedback:
    question: str
    chunk_id: str
    source: str
    page_start: int
    page_end: int
    rank: int
    score: int  # +1 or -1 (0 is never persisted)
    model_name: str
    timestamp: str


def record_feedback(
    path: Path,
    question: str,
    chunk_id: str,
    source: str,
    page_start: int,
    page_end: int,
    rank: int,
    score: int,
    model_name: str,
) -> None:
    if score not in (-1, 1):
        raise ValueError(f"score must be +1 or -1 to be recorded, got {score}")

    entry = Feedback(
        question=question,
        chunk_id=chunk_id,
        source=source,
        page_start=page_start,
        page_end=page_end,
        rank=rank,
        score=score,
        model_name=model_name,
        timestamp=datetime.now(timezone.utc).isoformat(),
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(asdict(entry)) + "\n")


def load_feedback(path: Path) -> list[Feedback]:
    if not path.exists():
        return []
    entries = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                entries.append(Feedback(**json.loads(line)))
    return entries
