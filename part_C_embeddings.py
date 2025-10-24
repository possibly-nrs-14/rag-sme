# part_C_embeddings.py
import os
import time
import hashlib
import numpy as np
import orjson
from sentence_transformers import SentenceTransformer

class TextEmbedder:
    def __init__(self, model=None, batch_size=128):
        self.model_name = model or "sentence-transformers/all-mpnet-base-v2"
        self.batch_size = batch_size
        self.backend = SentenceTransformer(self.model_name)
        self.dim = int(self.backend.get_sentence_embedding_dimension())

    def embed(self, texts):
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)

        if self.backend is not None:
            vecs = self.backend.encode(
                texts,
                batch_size=self.batch_size,
                normalize_embeddings=True,
                convert_to_numpy=True,
            )
            return vecs.astype(np.float32)

        # Deterministic hashing fallback (not semantic, but stable)
        dim = self.dim
        out = np.zeros((len(texts), dim), dtype=np.float32)
        for i, t in enumerate(texts):
            h = hashlib.sha256(t.encode("utf-8")).digest()
            for j, b in enumerate(h):
                out[i, j % dim] += (b - 127.5) / 128.0
            n = np.linalg.norm(out[i])
            if n > 0:
                out[i] /= n
        return out


def _write_jsonl(path, rows):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        for r in rows:
            f.write(orjson.dumps(r, option=orjson.OPT_APPEND_NEWLINE))


def save_document_graph(doc_id, basename, tokens, rows, out_dir, model=None):
    embedder = TextEmbedder(model=model)
    texts = [r["text"] for r in rows]
    embs = embedder.embed(texts)

    now = int(time.time())
    stem = os.path.splitext(basename)[0]

    doc_node_id = f"{doc_id}__doc"
    doc_node = {
        "node_id": doc_node_id,
        "node_type": "document",
        "doc_id": doc_id,
        "source_basename": basename,
        "parent_id": None,
        "granularity_tokens": None,
        "position": None,
        "embedding": None,
        "created_at": now,
    }

    chunk_nodes = []
    edges = []

    for i, r in enumerate(rows):
        node_id = r["chunk_id"]
        vec = embs[i].tolist()

        chunk_nodes.append({
            "node_id": node_id,
            "node_type": "chunk",
            "doc_id": doc_id,
            "parent_id": doc_node_id,
            "granularity_tokens": tokens,
            "position": r["position"],
            "embedding": vec,
            "created_at": r["created_at"],
        })

        edges.append({"src_id": node_id, "dst_id": doc_node_id, "edge_type": "child_of"})
        if r.get("prev_chunk_id"):
            edges.append({"src_id": r["prev_chunk_id"], "dst_id": node_id, "edge_type": "next"})
        if r.get("next_chunk_id"):
            edges.append({"src_id": node_id, "dst_id": r["next_chunk_id"], "edge_type": "next"})

    nodes_path = os.path.join(out_dir, "graph", f"{tokens}_tokens_{stem}__nodes.jsonl")
    edges_path = os.path.join(out_dir, "graph", f"{tokens}_tokens_{stem}__edges.jsonl")

    _write_jsonl(nodes_path, [doc_node] + chunk_nodes)
    _write_jsonl(edges_path, edges)

    return {"nodes_path": nodes_path, "edges_path": edges_path}


def save_chunk_embeddings_only(rows, out_path, model=None):
    embedder = TextEmbedder(model=model)
    texts = [r["text"] for r in rows]
    embs = embedder.embed(texts)

    out = []
    for i, r in enumerate(rows):
        out.append({
            "chunk_id": r["chunk_id"],
            "doc_id": r["doc_id"],
            "position": r["position"],
            "embedding": embs[i].tolist(),
        })

    _write_jsonl(out_path, out)
    return out_path