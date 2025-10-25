# Nervous System SME

## Directory layout

```
.
├─ data/                      # put your PDFs here
├─ artifacts/                 # all outputs land here (clean text, chunks, graph, logs)
├─ metadata/                  # file metadata from Part A
├─ part_A_collection.py       # file discovery & metadata
├─ part_B_preprocessing.py    # parsing, cleaning, chunking, de-dup, graph build
├─ part_C_embeddings.py       # embedding + graph writers
├─ rerank.py                  # retrieval + (optional) reranking
└─ run.py                     # end-to-end pipeline launcher
```

---

## Quick start

### 1) Environment

```bash
python -m venv .venv
.\.venv\Scripts\activate

python -m pip install --upgrade pip
pip install pymupdf datasketch tqdm orjson numpy faiss-cpu sentence-transformers
# Optional (for reranking):
pip install FlagEmbedding  # and a torch build (CPU or CUDA) if not already installed
```

### 2) Put PDFs in `./data`

Place your textbooks in `./data/`. E.g.:

```
./data/a_textbook_of_neuroanatomy.pdf
./data/barr_human_nervous_system.pdf
```

### 3) Run the pipeline

```bash
python run.py
```

This executes Part B’s `run_batch(...)` with sane defaults (granularities 2048/512/128 and 64-token overlap). 

Outputs appear in `./artifacts/` (see “Output Structure” below).

### 4) Try retrieval (with optional reranking)

```bash
python rerank.py
```

By default the sample at the bottom does a quick query. You can use `use_reranker=True` to enable BGE/CrossEncoder reranking (details below). 

---

## Script details

### `part_A_collection.py` — Input discovery & metadata

* Scans `./data/**` for supported documents (`pdf`, `docx`, `pptx`, `txt`, `md`).
* Writes `./metadata/file_metadata.json` and prints a summary so you can sanity-check inputs. 

Run it (optional):

```bash
python part_A_collection.py
```

---

### `part_B_preprocessing.py` — Parse ➜ Clean ➜ Chunk ➜ De-dup ➜ Graph

* **Block extraction with PyMuPDF** (`get_text("blocks")`) and **column-aware ordering** (left→right columns).
* **Chapter gating / front-matter skip** and **tail trimming** (e.g., basic `index` heuristics).
* **Sanitization** (neutralizes prompt-injection patterns) + **whitespace normalization** (delegates to `helpers.py`).
* **Paragraph-aware chunking** with overlapping windows, recursive fallback for long paras.
* **MinHash LSH** de-duplication to reduce near duplicates.
* **Writes**:
  * Cleaned text → `artifacts/clean/<book>__clean.txt`
  * Chunk JSONL per granularity → `artifacts/chunks/<tokens>_tokens_<book>.jsonl`
* **Also builds embeddings + a document graph** for each granularity **in-process** using Part C’s `save_document_graph(...)` (nodes/edges files).
* Finally, the `run_batch` function is used for batch ingestion of all documents in the data directory, and also does comprehensive logging, which is saved to a file at the end. (BONUS)


---

### `helpers.py` — Normalization & injection hardening 

* `normalize_spaces`: consistent whitespace/newline handling.
* `sanitize_for_injection`: strips control chars and replaces common instruction-like phrases. This is used while cleaning blocks and again before final chunk emission to avoid unwelcome downstream prompt-injections. (BONUS)

---

### `part_C_embeddings.py` — Embedding utilities & graph writer

* Lightweight wrapper around **SentenceTransformers** for text embedding.
* `save_document_graph(...)`:

  * Embeds each chunk and writes a **document node** and **chunk nodes** plus sequential/parent edges to:

    * `artifacts/graph/graph/<tokens>_tokens_<stem>__nodes.jsonl`
    * `artifacts/graph/graph/<tokens>_tokens_<stem>__edges.jsonl`
* `save_chunk_embeddings_only(...)`: optional per-chunk embeddings JSONL (if you want a flat vector file instead of the graph).
  This module is called **from Part B** after chunking, so you get embeddings/graph in one pass. 

---

### `rerank.py` — Retrieval + optional cross-encoder reranking (BONUS)

* Loads vectors/metadata from **either** `artifacts/embeddings/*__emb.jsonl` or **graph chunk nodes** under `artifacts/graph/**/__nodes.jsonl`.
* Builds a FAISS **inner product** index over L2-normalized vecs (equivalent to cosine).
* Encodes queries via `SentenceTransformer` and runs vector search.
* Optional **cross-encoder rerank**:
  * First choice: `FlagEmbedding.FlagReranker` (`BAAI/bge-reranker-base`), with auto FP16 on GPU, FP32 on CPU.
  * Fallback: `sentence-transformers` `CrossEncoder` (MiniLM) if FlagEmbedding isn’t available.
* Returns top-k results with both vector and (optionally) rerank scores.
  You can toggle reranking in code via `use_reranker=True/False`. 

---

### `run.py` — Runner script

Imports Part B’s `run_batch(...)` and executes the whole preprocessing + chunking + graph-embedding flow with default parameters and locations. Run it with:

```bash
python run.py
```

It expects data in `./data`, writes outputs to `./artifacts`. 

---

## Output Structure

After running `python run.py`:

```
artifacts/
  clean/
    <book>__clean.txt                 # normalized, sanitized text
  chunks/
    2048_tokens_<book>.jsonl          # per-chunk text + metadata
    512_tokens_<book>.jsonl
    128_tokens_<book>.jsonl
  graph/
    graph/
      2048_tokens_<stem>__nodes.jsonl # doc node + chunk nodes (with embeddings)
      2048_tokens_<stem>__edges.jsonl # child_of + next edges
      512_tokens_...                  # same at other granularities
      128_tokens_...
  logs/
    ingestion_<ts>.log
```

Then run `python rerank.py` to test the reranker.  

---

## Other

Details on data and outputs can be found in `data_and_outputs.pdf`. The log has also been uploaded (`ingestion_1761393988.log`)



