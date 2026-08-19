"""Build and query a FAISS index over embedded chunks.

Vectors are normalized and indexed with inner product, which is
equivalent to cosine similarity for unit vectors.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import faiss
import numpy as np

from rag.chunk import Chunk
from rag.embed import DEFAULT_MODEL, Embedder


@dataclass
class SearchResult:
    chunk: Chunk
    score: float


class ChunkIndex:
    def __init__(self, index: faiss.Index, chunks: list[Chunk], model_name: str):
        self.index = index
        self.chunks = chunks
        self.model_name = model_name

    @classmethod
    def build(cls, chunks: list[Chunk], embedder: Embedder) -> "ChunkIndex":
        vectors = embedder.embed_passages([c.text for c in chunks])
        dim = vectors.shape[1]
        index = faiss.IndexFlatIP(dim)
        index.add(vectors.astype(np.float32))
        return cls(index, chunks, embedder.model_name)

    def save(self, out_dir: Path) -> None:
        out_dir.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.index, str(out_dir / "index.faiss"))
        with (out_dir / "chunks.jsonl").open("w", encoding="utf-8") as f:
            for c in self.chunks:
                f.write(json.dumps(c.__dict__) + "\n")
        with (out_dir / "meta.json").open("w", encoding="utf-8") as f:
            json.dump({"model_name": self.model_name}, f)

    @classmethod
    def load(cls, index_dir: Path) -> "ChunkIndex":
        index = faiss.read_index(str(index_dir / "index.faiss"))
        chunks = []
        with (index_dir / "chunks.jsonl").open(encoding="utf-8") as f:
            for line in f:
                chunks.append(Chunk(**json.loads(line)))
        meta = json.loads((index_dir / "meta.json").read_text())
        return cls(index, chunks, meta["model_name"])

    def search(self, query_vector: np.ndarray, top_k: int = 5) -> list[SearchResult]:
        scores, indices = self.index.search(
            query_vector.reshape(1, -1).astype(np.float32), top_k
        )
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:
                continue
            results.append(SearchResult(chunk=self.chunks[idx], score=float(score)))
        return results


if __name__ == "__main__":
    import typer

    def main(
        chunks_path: Path = Path("data/processed/chunks.jsonl"),
        out_dir: Path = Path("data/index"),
        model: str = DEFAULT_MODEL,
    ):
        chunks = []
        with chunks_path.open(encoding="utf-8") as f:
            for line in f:
                chunks.append(Chunk(**json.loads(line)))
        if not chunks:
            raise SystemExit(f"No chunks found in {chunks_path}")

        print(f"Embedding {len(chunks)} chunks with {model} ...")
        embedder = Embedder(model)
        index = ChunkIndex.build(chunks, embedder)
        index.save(out_dir)
        print(f"Saved index -> {out_dir}")

    typer.run(main)
