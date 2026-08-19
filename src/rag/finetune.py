"""Fine-tune the bi-encoder on (question, passage) pairs.

Input format: JSONL with at least "question" and "passage" fields (see
data/finetune_pairs.jsonl for the expected shape). "passage" must be the
verbatim text of an already-embedded chunk; "question" is the synthetic
or user-collected anchor for it.

Uses MultipleNegativesRankingLoss (in-batch negatives): every other
passage in the same training batch acts as a negative for a given
question, so no explicit negatives are required to get started.
"""

from __future__ import annotations

import json
import random
from pathlib import Path

from datasets import Dataset
from sentence_transformers import (
    SentenceTransformer,
    SentenceTransformerTrainer,
    SentenceTransformerTrainingArguments,
)
from sentence_transformers.sentence_transformer.losses import MultipleNegativesRankingLoss

from rag.embed import DEFAULT_MODEL, QUERY_PREFIX

DEFAULT_PAIRS_PATH = Path("data/finetune_pairs.jsonl")
DEFAULT_OUTPUT_DIR = Path("data/models/bge-base-finetuned")

# Below this many pairs there isn't enough data for a meaningful held-out
# eval split, so we train on everything and skip eval.
MIN_PAIRS_FOR_EVAL_SPLIT = 10
EVAL_FRACTION = 0.1


def load_pairs(path: Path) -> list[dict]:
    pairs = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            if "<" in row["question"]:  # unfilled template placeholder
                continue
            pairs.append({"question": row["question"], "passage": row["passage"]})
    return pairs


def build_dataset(pairs: list[dict]) -> Dataset:
    return Dataset.from_dict(
        {
            "anchor": [QUERY_PREFIX + p["question"] for p in pairs],
            "positive": [p["passage"] for p in pairs],
        }
    )


def run(
    pairs_path: Path = DEFAULT_PAIRS_PATH,
    base_model: str = DEFAULT_MODEL,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    epochs: float = 3.0,
    batch_size: int = 16,
    learning_rate: float = 2e-5,
    seed: int = 42,
) -> Path:
    pairs = load_pairs(pairs_path)
    if len(pairs) < 2:
        raise ValueError(
            f"Need at least 2 (question, passage) pairs to train, found {len(pairs)} in {pairs_path}"
        )

    random.Random(seed).shuffle(pairs)
    n_eval = int(len(pairs) * EVAL_FRACTION) if len(pairs) >= MIN_PAIRS_FOR_EVAL_SPLIT else 0
    eval_pairs, train_pairs = pairs[:n_eval], pairs[n_eval:]

    train_dataset = build_dataset(train_pairs)
    eval_dataset = build_dataset(eval_pairs) if eval_pairs else None
    effective_batch_size = min(batch_size, len(train_pairs))

    model = SentenceTransformer(base_model)
    loss = MultipleNegativesRankingLoss(model)

    args = SentenceTransformerTrainingArguments(
        output_dir=str(output_dir / "checkpoints"),
        num_train_epochs=epochs,
        per_device_train_batch_size=effective_batch_size,
        per_device_eval_batch_size=effective_batch_size,
        learning_rate=learning_rate,
        warmup_steps=0.1,  # fraction of training steps, per current transformers guidance
        eval_strategy="epoch" if eval_dataset is not None else "no",
        save_strategy="epoch",
        save_total_limit=1,
        logging_steps=max(1, len(train_pairs) // effective_batch_size),
        seed=seed,
    )

    trainer = SentenceTransformerTrainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        loss=loss,
    )
    trainer.train()

    final_dir = output_dir / "final"
    model.save(str(final_dir))
    return final_dir


if __name__ == "__main__":
    import typer

    def main(
        pairs_path: Path = DEFAULT_PAIRS_PATH,
        base_model: str = DEFAULT_MODEL,
        output_dir: Path = DEFAULT_OUTPUT_DIR,
        epochs: float = 3.0,
        batch_size: int = 16,
        learning_rate: float = 2e-5,
    ):
        final_dir = run(
            pairs_path=pairs_path,
            base_model=base_model,
            output_dir=output_dir,
            epochs=epochs,
            batch_size=batch_size,
            learning_rate=learning_rate,
        )
        print(f"Saved fine-tuned model -> {final_dir}")
        print(
            f"To use it: uv run python -m rag.index --model {final_dir} "
            "(re-embeds the corpus with the new weights)"
        )

    typer.run(main)
