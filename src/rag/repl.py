"""Interactive retrieval loop: load the model and index once, then answer
many questions in a row without paying the ~6s model-load cost per query.
"""

from __future__ import annotations

from pathlib import Path

from rich.console import Console

from rag.embed import Embedder
from rag.feedback import DEFAULT_FEEDBACK_PATH
from rag.index import ChunkIndex
from rag.retrieve import collect_feedback, print_results, query

console = Console()


def run(
    index_dir: Path,
    top_k: int,
    feedback: bool,
    feedback_path: Path,
) -> None:
    console.print(f"[dim]Loading index from {index_dir} ...[/dim]")
    index = ChunkIndex.load(index_dir)
    console.print(f"[dim]Loading model {index.model_name} ...[/dim]")
    embedder = Embedder(index.model_name)
    console.print(
        f"[bold]Ready.[/bold] {len(index.chunks)} chunks indexed. "
        "Type a question, or 'quit'/'exit' (or Ctrl-D) to stop.\n"
    )

    while True:
        try:
            question = input("> ").strip()
        except EOFError:
            console.print()
            break

        if not question:
            continue
        if question.lower() in ("quit", "exit"):
            break

        results = query(index, embedder, question, top_k=top_k)
        print_results(results)

        if feedback:
            collect_feedback(index, question, results, feedback_path)
        console.print()


if __name__ == "__main__":
    import typer

    def main(
        index_dir: Path = Path("data/index"),
        top_k: int = 5,
        feedback: bool = False,
        feedback_path: Path = DEFAULT_FEEDBACK_PATH,
    ):
        run(index_dir, top_k, feedback, feedback_path)

    typer.run(main)
