"""Evaluate retrieval quality against a labeled question set.

Eval set format (JSONL), one object per line:
    {"question": "...", "source": "textbook.pdf", "page": 42}

`page` is the page in the source PDF a human has confirmed contains the
answer. A retrieved chunk counts as a hit if it's from the same source
and its page range covers that page.
"""

from __future__ import annotations

import json
from pathlib import Path

from rag.embed import Embedder
from rag.index import ChunkIndex
from rag.retrieve import query


def is_hit(result_chunk, source: str, page: int) -> bool:
    return result_chunk.source == source and result_chunk.page_start <= page <= result_chunk.page_end


def evaluate(index: ChunkIndex, embedder: Embedder, eval_path: Path, top_k: int = 10):
    examples = [json.loads(line) for line in eval_path.read_text().splitlines() if line.strip()]

    recall_at_1 = 0
    recall_at_5 = 0
    recall_at_10 = 0
    reciprocal_ranks = []

    for ex in examples:
        results = query(index, embedder, ex["question"], top_k=top_k)
        rank = None
        for i, r in enumerate(results, start=1):
            if is_hit(r.chunk, ex["source"], ex["page"]):
                rank = i
                break
        reciprocal_ranks.append(1 / rank if rank else 0.0)
        if rank and rank <= 1:
            recall_at_1 += 1
        if rank and rank <= 5:
            recall_at_5 += 1
        if rank and rank <= 10:
            recall_at_10 += 1

    n = len(examples)
    return {
        "n": n,
        "recall@1": recall_at_1 / n,
        "recall@5": recall_at_5 / n,
        "recall@10": recall_at_10 / n,
        "mrr": sum(reciprocal_ranks) / n,
    }


if __name__ == "__main__":
    import typer

    def main(
        eval_path: Path = Path("data/eval_questions.jsonl"),
        index_dir: Path = Path("data/index"),
        top_k: int = 10,
    ):
        index = ChunkIndex.load(index_dir)
        embedder = Embedder(index.model_name)
        metrics = evaluate(index, embedder, eval_path, top_k=top_k)
        for k, v in metrics.items():
            print(f"{k}: {v}")

    typer.run(main)
