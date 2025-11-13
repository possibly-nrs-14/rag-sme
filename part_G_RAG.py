# Part_G_RAG.py
import os, glob, json
import numpy as np
import orjson as _oj
import faiss
import torch

try:
    from FlagEmbedding import FlagReranker
    _HAS_FLAG = True
except Exception:
    FlagReranker = None
    _HAS_FLAG = False

try:
    from sentence_transformers import SentenceTransformer, CrossEncoder
    _HAS_ST = True
    _HAS_XE = True
except Exception:
    SentenceTransformer = None
    CrossEncoder = None
    _HAS_ST = False
    _HAS_XE = False

from part_C_embeddings import DEFAULT_EMBED_MODEL

def _json_lines(path):
    try:
        with open(path, "rb") as f:
            for line in f:
                if line.strip():
                    yield _oj.loads(line)
    except Exception:
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    yield json.loads(line)

def _as_float32(x):
    x = np.asarray(x)
    if x.dtype != np.float32:
        x = x.astype("float32")
    return x

def _norm_rows(x):
    n = np.linalg.norm(x, axis=1, keepdims=True) + 1e-12
    return x / n

def _field(row, *candidates, default=None):
    for k in candidates:
        if k in row and row[k] is not None:
            return row[k]
    meta = row.get("metadata") or {}
    for k in candidates:
        if k in meta and meta[k] is not None:
            return meta[k]
    return default


class EmbeddingIndex:
    def __init__(self, vectors, metas):
        self.vectors = _norm_rows(_as_float32(vectors))
        self.metas = metas
        d = self.vectors.shape[1]
        self.index = faiss.IndexFlatIP(d)
        self.index.add(self.vectors)

    @staticmethod
    def from_files(patterns=("./artifacts/embeddings/*__emb.jsonl",
                             "./artifacts/graph/*__nodes.jsonl")):
        files = []
        for pat in patterns:
            files.extend(sorted(glob.glob(pat, recursive=True)))
        if not files:
            raise RuntimeError("No embedding/graph files found under ./artifacts.")

        doc_name = {}
        for fp in files:
            for row in _json_lines(fp):
                node_type = (_field(row, "type", "node_type", default="") or "").lower()
                if node_type in ("document", "doc", "node_document", "document_node"):
                    did = _field(row, "doc_id", "id", "node_id")
                    if not did:
                        continue
                    base = _field(row, "source_basename", "basename", "title")
                    if not base:
                        sp = _field(row, "source_path")
                        if sp:
                            base = os.path.basename(sp)
                    if base:
                        doc_name[did] = base

        metas, vecs = [], []
        for fp in files:
            for row in _json_lines(fp):
                v, t, did = None, "", None

                if ("vector" in row) or ("embedding" in row) or ("emb" in row):
                    v = _field(row, "vector", "embedding", "emb")
                    t = _field(row, "text", "chunk_text", "content", default="")
                    did = _field(row, "doc_id")
                    source_basename = _field(row, "source_basename", "basename")
                    if not source_basename:
                        sp = _field(row, "source_path")
                        if sp:
                            source_basename = os.path.basename(sp)
                else:
                    node_type = (_field(row, "type", "node_type", default="") or "").lower()
                    if node_type not in ("chunk", "node_chunk"):
                        continue
                    v = _field(row, "embedding", "vector")
                    t = _field(row, "text", "content", default="")
                    did = _field(row, "doc_id")
                    source_basename = _field(row, "source_basename", "basename")
                    if not source_basename and did and did in doc_name:
                        source_basename = doc_name[did]

                if v is None:
                    continue

                metas.append({
                    "chunk_id": _field(row, "chunk_id", "node_id", "id"),
                    "doc_id": did,
                    "source_basename": source_basename,
                    "granularity_tokens": _field(row, "granularity_tokens", "tokens"),
                    "position": _field(row, "position", "idx"),
                    "text": t,
                    "score_vec": None,
                    "score_rerank": None,
                })
                vecs.append(np.array(v, dtype="float32"))

        if not vecs:
            raise RuntimeError("No vectors found in embeddings or graph nodes.")
        X = np.vstack(vecs)
        return EmbeddingIndex(X, metas)

    def vector_search(self, query_vector, top_k=50):
        qv = query_vector.reshape(1, -1)
        D, I = self.index.search(qv, top_k)
        return D[0], I[0]


class QueryEncoder:
    def __init__(self, model_name=DEFAULT_EMBED_MODEL, device=None):
        if not _HAS_ST:
            raise RuntimeError("sentence-transformers not installed")
        self.model = SentenceTransformer(model_name, device=device)

    def encode(self, texts, normalize=True, batch_size=32):
        v = self.model.encode(texts, batch_size=batch_size, normalize_embeddings=normalize)
        return _as_float32(v)

    def encode_one(self, text, normalize=True):
        return self.encode([text], normalize=normalize)[0]


class BGEReranker:
    def __init__(self, model_name="BAAI/bge-reranker-base", device=None):
        self.available = False
        self.model = None
        self.fallback = None
        try:
            if device is None:
                device = "cuda" if torch.cuda.is_available() else "cpu"
            use_fp16 = (device == "cuda")
            if _HAS_FLAG:
                self.model = FlagReranker(model_name, use_fp16=use_fp16, device=device)
                self.available = True
            elif _HAS_XE and CrossEncoder is not None:
                self.fallback = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2", device=device)
                self.available = True
        except Exception:
            self.available = False
            self.model = None
            self.fallback = None

    def rerank(self, query, texts, batch_size=16):
        if not self.available or not texts:
            return None
        texts = [(t or "") for t in texts]
        try:
            pairs = [[query, t] for t in texts]
            if self.model is not None:
                return self.model.compute_score(pairs, batch_size=batch_size)
            if self.fallback is not None:
                return self.fallback.predict(pairs, batch_size=batch_size).tolist()
        except Exception:
            return None


class SearchPipeline:
    def __init__(self, index, encoder, reranker=None):
        self.index = index
        self.encoder = encoder
        self.reranker = reranker

    def retrieve(self, query, k=50):
        qv = self.encoder.encode_one(query, normalize=True)
        D, I = self.index.vector_search(qv, top_k=k)
        out = []
        for j, idx in enumerate(I):
            m = dict(self.index.metas[idx])
            m["score_vec"] = float(D[j])
            out.append(m)
        return out

    def search(self, query, top_k=10, candidates=100, use_reranker=False, rerank_batch_size=32):
        qv = self.encoder.encode_one(query, normalize=True)
        D, I = self.index.vector_search(qv, top_k=max(top_k, candidates))
        for j, idx in enumerate(I):
            self.index.metas[idx]["score_vec"] = float(D[j])

        cand_idx = list(I[:candidates])
        if use_reranker and self.reranker is not None and self.reranker.available:
            texts = [self.index.metas[i]["text"] for i in cand_idx]
            scores = self.reranker.rerank(query, texts, batch_size=rerank_batch_size)
            if scores is not None:
                for j, idx in enumerate(cand_idx):
                    self.index.metas[idx]["score_rerank"] = float(scores[j])
                cand_idx.sort(key=lambda k: (self.index.metas[k]["score_rerank"],
                                             self.index.metas[k]["score_vec"]), reverse=True)
            else:
                cand_idx.sort(key=lambda k: self.index.metas[k]["score_vec"], reverse=True)
        else:
            cand_idx.sort(key=lambda k: self.index.metas[k]["score_vec"], reverse=True)

        final = []
        for idx in cand_idx[:top_k]:
            final.append(dict(self.index.metas[idx]))
        return final
