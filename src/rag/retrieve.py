"""Query the built index: given a question, return the passages most
likely to contain the answer."""

from __future__ import annotations

from pathlib import Path

from rich.console import Console
from rich.panel import Panel

from rag.embed import Embedder
from rag.feedback import DEFAULT_FEEDBACK_PATH, record_feedback
from rag.index import ChunkIndex

console = Console()


def query(index: ChunkIndex, embedder: Embedder, question: str, top_k: int = 5):
    query_vector = embedder.embed_queries([question])[0]
    return index.search(query_vector, top_k=top_k)


def print_results(results) -> None:
    for rank, r in enumerate(results, start=1):
        pages = (
            f"p.{r.chunk.page_start}"
            if r.chunk.page_start == r.chunk.page_end
            else f"pp.{r.chunk.page_start}-{r.chunk.page_end}"
        )
        title = f"#{rank}  score={r.score:.3f}  {r.chunk.source} ({pages})"
        console.print(Panel(r.chunk.text, title=title, expand=False))


def collect_feedback(
    index: ChunkIndex,
    question: str,
    results,
    feedback_path: Path = DEFAULT_FEEDBACK_PATH,
) -> None:
    console.print(
        "\n[bold]Rate each result[/bold]: +1 = answered the question, "
        "-1 = did not, Enter = skip"
    )
    for rank, r in enumerate(results, start=1):
        raw = input(f"  #{rank} [{r.chunk.source}]: ").strip()
        if raw == "":
            continue
        if raw not in ("+1", "1", "-1"):
            console.print(f"  [dim]ignored '{raw}' (expected +1, -1, or blank)[/dim]")
            continue
        score = 1 if raw in ("+1", "1") else -1
        record_feedback(
            feedback_path,
            question=question,
            chunk_id=r.chunk.chunk_id,
            source=r.chunk.source,
            page_start=r.chunk.page_start,
            page_end=r.chunk.page_end,
            rank=rank,
            score=score,
            model_name=index.model_name,
        )
    console.print(f"[dim]Feedback saved -> {feedback_path}[/dim]")


if __name__ == "__main__":
    import typer

    def main(
        question: str,
        index_dir: Path = Path("data/index"),
        top_k: int = 5,
        feedback: bool = False,
        feedback_path: Path = DEFAULT_FEEDBACK_PATH,
    ):
        index = ChunkIndex.load(index_dir)
        embedder = Embedder(index.model_name)
        results = query(index, embedder, question, top_k=top_k)
        print_results(results)

        if feedback:
            collect_feedback(index, question, results, feedback_path)

    typer.run(main)
