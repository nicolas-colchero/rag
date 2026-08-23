# Retrieval System

This project consists in building a Retrieval system capable of finding the answer to a question inside a collection of books.  
The way it works is quite simple: First, the book is split into passages (actually, it is split into overlapping chunks, each chunk consists of 220 words, with 50 overlapping words (at the start and the end). This way we ensure each whole sentence is likely contained in some chunk) and each passage is encoded using an encoder model (by default, using the BAAI BGE [[1]](#1) model).  
The encodings are stored in a FAISS [[2]](#2) index. These steps are done once, before usage.

Then, after a query is received, the index is loaded and the query is encoded using the same encoder model. The query's encoding is then compared to each passage encoding (via dot product), to detect similarities. The passages corresponding to the top 5 highest dot products are returned since they share information with the query (and hopefully answer the query's answer).

## Fine-tunning

Two possible ways of fine-tunning the model have been implemented:

- Finetunning via (query, passage) pairs: The model creates an encoding for both the query and the passage for each pair on the training batch (of size N). Then, it computes a $N\times N$ matrix containing the cross product of each passage encoding and each query encoding. The diagonal of this matrix are the correct matches, while every other score is treated as a negative. Then, we use softmax and cross-entropy in each row.  
In short, each query is compared to each passage in the batch. Similarities with respect to the passage in the same pair are encouraged, while similarities with respect to any other passage are discouraged.
- Finetunning via feedback: The model can record user's feedback during ussage. The fine-tunning cannot occur online (during ussage) since changing the model requires updating the encodings of the passages. Instead, a feedback file is maintained with the feedback obtained from the user. This feedback can be converted into a fine-tunning file (positive feedbacks are converted into fine-tunning pairs), thus we can employ the previous fine-tunning method. In the current implementation, negative feedback is discarded since the fine-tunning method does not allow for hard negatives (as explained before, it uses all examples in the batch (other than the positive example) as soft negatives).

## Evaluation

Evalution can be performed via a file containing (question, page, book) tuples. During evaluation, a question is considered correctly solved at k if passages from the correct page appear in the top k answers of the question. Recall is the metric of choice since the evaluation dataset only contains positive classes (all questions are correctly solved by the correct page of the correct book).

## Manual

All commands run via `uv run python -m rag.<module>`; add `--help` to any of them for the full flag list.

**1. Setup.** Install dependencies with `uv sync`. `.djvu`/`.djv` books also need the `djvulibre` CLI tools installed system-wide (e.g. `pacman -S djvulibre`).

**2. Build the index**, one step at a time. Drop your PDFs/DjVu files into `data/raw_books/` first.
```bash
uv run python -m rag.ingest    # extract text -> data/processed/pages.jsonl
uv run python -m rag.chunk     # split into overlapping chunks -> data/processed/chunks.jsonl
uv run python -m rag.index     # embed chunks and build the FAISS index -> data/index/
```

**3. Ask questions.**
```bash
uv run python -m rag.retrieve "your question here"   # one-off query
uv run python -m rag.repl                             # interactive loop (loads the model once)
```
Add `--top-k N` to change how many passages come back.

**4. Rate results (optional).** Pass `--feedback` to either command above and you'll be prompted to rate each result `+1` (answered the question) or `-1` (didn't), or hit Enter to skip. Ratings are logged to `data/feedback.jsonl`.

**5. Improve the model with your feedback (optional).**
```bash
uv run python -m rag.build_pairs   # turn +1-rated feedback into training pairs -> data/finetune_pairs.jsonl
uv run python -m rag.finetune      # fine-tune the encoder on those pairs -> data/models/bge-base-finetuned/final
uv run python -m rag.index --model data/models/bge-base-finetuned/final   # re-embed the corpus with the new weights
```

**6. Check retrieval quality.** Given a labeled question set (one JSON object per line: `{"question": "...", "source": "book.pdf", "page": 42}`, e.g. `data/eval_questions.jsonl`):
```bash
uv run python -m rag.eval
```
Reports recall@1/5/10 and MRR.

## Reference and names

<a id="1">[1]</a> : Beijing Academy of Artificial Intelligence Generalized Embedding  
<a id="2">[2]</a> : A Python library for efficient searching