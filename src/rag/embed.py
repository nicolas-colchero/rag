"""Embedding wrapper around a local sentence-transformers model.

Swapping models (or later, a fine-tuned checkpoint) only requires
changing DEFAULT_MODEL / passing --model — nothing downstream depends
on which model produced the vectors.
"""

from __future__ import annotations

import numpy as np
from sentence_transformers import SentenceTransformer

# BGE models are instruction-tuned: passages are embedded as-is, but
# queries need a prefix to align with how the model was trained.
DEFAULT_MODEL = "BAAI/bge-base-en-v1.5"
QUERY_PREFIX = "Represent this sentence for searching relevant passages: "


class Embedder:
    def __init__(self, model_name: str = DEFAULT_MODEL, device: str | None = None):
        self.model = SentenceTransformer(model_name, device=device)
        self.model_name = model_name

    def embed_passages(self, texts: list[str], batch_size: int = 64) -> np.ndarray:
        return self.model.encode(
            texts,
            batch_size=batch_size,
            normalize_embeddings=True,
            show_progress_bar=len(texts) > 100,
            convert_to_numpy=True,
        )

    def embed_queries(self, texts: list[str], batch_size: int = 64) -> np.ndarray:
        prefixed = [QUERY_PREFIX + t for t in texts]
        return self.model.encode(
            prefixed,
            batch_size=batch_size,
            normalize_embeddings=True,
            show_progress_bar=False,
            convert_to_numpy=True,
        )
