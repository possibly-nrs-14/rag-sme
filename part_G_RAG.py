# Part_G_RAG.py
import os, glob, json
import logging
import numpy as np
import orjson as _oj
import faiss
import torch
from pathlib import Path

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

try:
    from elasticsearch import Elasticsearch
    from elasticsearch.helpers import bulk as es_bulk
    _HAS_ES = True
except Exception:
    Elasticsearch = None
    _HAS_ES = False

from part_C_embeddings import DEFAULT_EMBED_MODEL

logger = logging.getLogger(__name__)

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


def load_elasticsearch_config(config_path=None):
    if config_path is None:
        # Try relative to current file
        current_dir = Path(__file__).parent
        config_path = current_dir / "config" / "elasticsearch.json"

        # If not found, try relative to working directory
        if not config_path.exists():
            config_path = Path("config") / "elasticsearch.json"
    else:
        config_path = Path(config_path)

    if not config_path.exists():
        logger.warning(f"Elasticsearch config not found at {config_path}")
        return None

    try:
        with open(config_path, 'r') as f:
            config = json.load(f)

        # Check if enabled
        if not config.get("enabled", True):
            logger.info("Elasticsearch disabled in config")
            return None

        return config
    except Exception as e:
        logger.error(f"Failed to load ES config: {e}")
        return None


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


class ElasticsearchIndex:
    def __init__(self, es_client=None, index_name="sme_chunks", config=None):
        if not _HAS_ES:
            raise RuntimeError("elasticsearch package not installed. Install with: pip install elasticsearch>=8.11.0")

        # Load config if provided as path or dict
        if config is None:
            config = load_elasticsearch_config()
        elif isinstance(config, (str, Path)):
            config = load_elasticsearch_config(config)

        if config is None:
            raise RuntimeError("No valid Elasticsearch configuration found")

        self.config = config
        self.index_name = config.get("index_name", index_name)

        # Create ES client if not provided
        if es_client is None:
            host = config.get("host", "localhost")
            port = config.get("port", 9200)
            scheme = config.get("scheme", "http")
            timeout = config.get("request_timeout", 30)

            # Elasticsearch 8.x uses a different initialization
            es_url = f"{scheme}://{host}:{port}"

            self.es = Elasticsearch(
                hosts=[es_url],
                request_timeout=timeout,
                max_retries=config.get("max_retries", 3),
                retry_on_timeout=True
            )
        else:
            self.es = es_client
            host = config.get("host", "localhost")
            port = config.get("port", 9200)
            scheme = config.get("scheme", "http")
            es_url = f"{scheme}://{host}:{port}"

        # Test connection using info() instead of ping() for ES 8.x+ compatibility
        try:
            info = self.es.info()
            cluster_name = info.get('cluster_name', 'unknown')
            version = info.get('version', {}).get('number', 'unknown')
            logger.info(f"Connected to Elasticsearch at {es_url} (cluster: {cluster_name}, version: {version})")
        except Exception as e:
            raise RuntimeError(f"Cannot connect to Elasticsearch at {es_url}: {e}")

    def create_index(self, dims=768, delete_if_exists=False):
        """
        Create Elasticsearch index with mapping for dense vectors and BM25.

        Args:
            dims: Dimension of embedding vectors
            delete_if_exists: If True, delete existing index before creating
        """
        if delete_if_exists and self.es.indices.exists(index=self.index_name):
            self.es.indices.delete(index=self.index_name)
            logger.info(f"Deleted existing index: {self.index_name}")

        if self.es.indices.exists(index=self.index_name):
            logger.info(f"Index {self.index_name} already exists, skipping creation")
            return

        settings = self.config.get("settings", {
            "number_of_shards": 1,
            "number_of_replicas": 0
        })

        similarity = self.config.get("similarity", "cosine")

        mapping = {
            "properties": {
                "chunk_id": {"type": "keyword"},
                "doc_id": {"type": "keyword"},
                "source_basename": {"type": "keyword"},
                "granularity_tokens": {"type": "integer"},
                "position": {"type": "integer"},
                "n_tokens": {"type": "integer"},
                "text": {
                    "type": "text",
                    "analyzer": "english"
                },
                "embedding": {
                    "type": "dense_vector",
                    "dims": dims,
                    "index": True,
                    "similarity": similarity
                },
                "parent_doc_hash": {"type": "keyword"},
                "source_path": {"type": "keyword"},
                "created_at": {"type": "long"}
            }
        }

        self.es.indices.create(
            index=self.index_name,
            settings=settings,
            mappings=mapping
        )
        logger.info(f"Created index: {self.index_name} with {dims}-dim vectors ({similarity} similarity)")

    def index_chunk(self, chunk_data):
        """
        Index a single chunk to Elasticsearch.

        Args:
            chunk_data: Dict with keys: chunk_id, text, embedding, and metadata
        """
        doc = {
            "_index": self.index_name,
            "_id": chunk_data.get("chunk_id"),
            "chunk_id": chunk_data.get("chunk_id"),
            "doc_id": chunk_data.get("doc_id"),
            "text": chunk_data.get("text", ""),
            "embedding": chunk_data.get("embedding"),
            "source_basename": chunk_data.get("source_basename"),
            "source_path": chunk_data.get("source_path"),
            "granularity_tokens": chunk_data.get("granularity_tokens"),
            "position": chunk_data.get("position"),
            "n_tokens": chunk_data.get("n_tokens"),
            "parent_doc_hash": chunk_data.get("parent_doc_hash"),
            "created_at": chunk_data.get("created_at")
        }

        self.es.index(index=self.index_name, id=doc["_id"], document=doc)

    def index_chunks_bulk(self, chunks, batch_size=None):
        """
        Bulk index multiple chunks to Elasticsearch.

        Args:
            chunks: List of chunk dicts
            batch_size: Number of documents per batch (from config if None)

        Returns:
            tuple: (success_count, failed_count)
        """
        if batch_size is None:
            batch_size = self.config.get("bulk_size", 500)

        actions = []
        for chunk in chunks:
            action = {
                "_index": self.index_name,
                "_id": chunk.get("chunk_id"),
                "_source": {
                    "chunk_id": chunk.get("chunk_id"),
                    "doc_id": chunk.get("doc_id"),
                    "text": chunk.get("text", ""),
                    "embedding": chunk.get("embedding"),
                    "source_basename": chunk.get("source_basename"),
                    "source_path": chunk.get("source_path"),
                    "granularity_tokens": chunk.get("granularity_tokens"),
                    "position": chunk.get("position"),
                    "n_tokens": chunk.get("n_tokens"),
                    "parent_doc_hash": chunk.get("parent_doc_hash"),
                    "created_at": chunk.get("created_at")
                }
            }
            actions.append(action)

        success, failed = es_bulk(self.es, actions, chunk_size=batch_size, raise_on_error=False)
        logger.info(f"Bulk indexed {success} chunks, {len(failed)} failed")
        return success, len(failed)

    def vector_search(self, query_vector, top_k=50, filters=None):
        """
        Perform kNN vector search.

        Args:
            query_vector: Query embedding vector (numpy array or list)
            top_k: Number of results to return
            filters: Optional dict of filters (e.g., {"source_basename": "book.pdf"})

        Returns:
            tuple: (scores, metadata_list)
        """
        query_vec = query_vector.tolist() if isinstance(query_vector, np.ndarray) else query_vector

        knn_query = {
            "field": "embedding",
            "query_vector": query_vec,
            "k": top_k,
            "num_candidates": max(top_k * 2, 100)
        }

        # Add filters if provided
        if filters:
            filter_clauses = []
            for key, value in filters.items():
                filter_clauses.append({"term": {key: value}})
            knn_query["filter"] = filter_clauses

        response = self.es.search(
            index=self.index_name,
            knn=knn_query,
            size=top_k,
            _source=["chunk_id", "doc_id", "text", "source_basename", "granularity_tokens", "position"]
        )

        scores = []
        metas = []
        for hit in response["hits"]["hits"]:
            scores.append(hit["_score"])
            meta = hit["_source"]
            meta["score_vec"] = hit["_score"]
            meta["score_rerank"] = None
            metas.append(meta)

        return np.array(scores), metas

    def bm25_search(self, query_text, top_k=50, filters=None):
        """
        Perform BM25 keyword search.

        Args:
            query_text: Query string
            top_k: Number of results to return
            filters: Optional dict of filters

        Returns:
            tuple: (scores, metadata_list)
        """
        query = {
            "match": {
                "text": {
                    "query": query_text,
                    "operator": "or"
                }
            }
        }

        # Add filters if provided
        if filters:
            bool_query = {
                "bool": {
                    "must": [query],
                    "filter": [{"term": {k: v}} for k, v in filters.items()]
                }
            }
            query = bool_query

        response = self.es.search(
            index=self.index_name,
            query=query,
            size=top_k,
            _source=["chunk_id", "doc_id", "text", "source_basename", "granularity_tokens", "position"]
        )

        scores = []
        metas = []
        for hit in response["hits"]["hits"]:
            scores.append(hit["_score"])
            meta = hit["_source"]
            meta["score_bm25"] = hit["_score"]
            meta["score_vec"] = None
            meta["score_rerank"] = None
            metas.append(meta)

        return np.array(scores), metas

    def hybrid_search(self, query_text, query_vector, top_k=50, alpha=0.5, filters=None):
        # Get candidates from both searches
        candidates_k = max(top_k * 2, 100)

        # Vector search
        vec_scores, vec_metas = self.vector_search(query_vector, top_k=candidates_k, filters=filters)
        vec_results = {m["chunk_id"]: (s, m) for s, m in zip(vec_scores, vec_metas)}

        # BM25 search
        bm25_scores, bm25_metas = self.bm25_search(query_text, top_k=candidates_k, filters=filters)
        bm25_results = {m["chunk_id"]: (s, m) for s, m in zip(bm25_scores, bm25_metas)}

        # Reciprocal Rank Fusion (RRF)
        rrf_scores = {}
        k_rrf = 60  # RRF constant

        # Add vector search ranks
        for rank, chunk_id in enumerate(vec_results.keys()):
            rrf_scores[chunk_id] = alpha / (k_rrf + rank + 1)

        # Add BM25 ranks
        for rank, chunk_id in enumerate(bm25_results.keys()):
            rrf_scores[chunk_id] = rrf_scores.get(chunk_id, 0) + (1 - alpha) / (k_rrf + rank + 1)

        # Sort by RRF score and get top-k
        sorted_chunks = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)[:top_k]

        # Collect results with metadata
        final_scores = []
        final_metas = []
        for chunk_id, rrf_score in sorted_chunks:
            # Prefer vector search metadata if available
            if chunk_id in vec_results:
                _, meta = vec_results[chunk_id]
            else:
                _, meta = bm25_results[chunk_id]

            # Add both scores
            if chunk_id in vec_results:
                meta["score_vec"] = vec_results[chunk_id][0]
            if chunk_id in bm25_results:
                meta["score_bm25"] = bm25_results[chunk_id][0]

            meta["score_hybrid"] = rrf_score

            final_scores.append(rrf_score)
            final_metas.append(meta)

        return np.array(final_scores), final_metas

    def get_document_count(self):
        """Get total number of indexed documents."""
        return self.es.count(index=self.index_name)["count"]


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
        """
        Unified search pipeline supporting both FAISS and Elasticsearch backends.

        Args:
            index: EmbeddingIndex (FAISS) or ElasticsearchIndex instance
            encoder: QueryEncoder instance
            reranker: Optional BGEReranker instance
        """
        self.index = index
        self.encoder = encoder
        self.reranker = reranker
        self.is_elasticsearch = isinstance(index, ElasticsearchIndex)

    def retrieve(self, query, k=50):
        """Legacy method for backward compatibility."""
        qv = self.encoder.encode_one(query, normalize=True)

        if self.is_elasticsearch:
            _, metas = self.index.vector_search(qv, top_k=k)
            return metas
        else:
            # FAISS backend
            D, I = self.index.vector_search(qv, top_k=k)
            out = []
            for j, idx in enumerate(I):
                m = dict(self.index.metas[idx])
                m["score_vec"] = float(D[j])
                out.append(m)
            return out

    def search(self, query, top_k=10, candidates=100, use_reranker=False,
               rerank_batch_size=32, search_mode="hybrid", hybrid_alpha=0.5, filters=None):
        """
        Perform search with optional reranking.

        Args:
            query: Query string
            top_k: Number of final results to return
            candidates: Number of candidates to retrieve before reranking
            use_reranker: Whether to use BGE reranker
            rerank_batch_size: Batch size for reranking
            search_mode: "vector", "bm25", or "hybrid" (default: "hybrid", only for Elasticsearch)
            hybrid_alpha: Weight for hybrid search (0=full BM25, 1=full vector, 0.5=balanced)
            filters: Optional dict of metadata filters (only for Elasticsearch)

        Returns:
            list: List of result dicts with metadata and scores
        """
        qv = self.encoder.encode_one(query, normalize=True)

        # Perform initial search based on backend and mode
        if self.is_elasticsearch:
            # Elasticsearch backend with multi-mode support
            candidates_k = max(top_k, candidates)

            if search_mode == "vector":
                _, results = self.index.vector_search(qv, top_k=candidates_k, filters=filters)
            elif search_mode == "bm25":
                _, results = self.index.bm25_search(query, top_k=candidates_k, filters=filters)
            elif search_mode == "hybrid":
                _, results = self.index.hybrid_search(query, qv, top_k=candidates_k,
                                                     alpha=hybrid_alpha, filters=filters)
            else:
                raise ValueError(f"Invalid search_mode: {search_mode}. Use 'vector', 'bm25', or 'hybrid'")

        else:
            # FAISS backend (vector-only)
            if search_mode != "vector" and search_mode != "hybrid":
                logger.warning(f"FAISS backend only supports vector search, ignoring search_mode='{search_mode}'")

            D, I = self.index.vector_search(qv, top_k=max(top_k, candidates))
            results = []
            for j, idx in enumerate(I[:candidates]):
                m = dict(self.index.metas[idx])
                m["score_vec"] = float(D[j])
                results.append(m)

        # Apply reranking if requested
        if use_reranker and self.reranker is not None and self.reranker.available and results:
            texts = [r["text"] for r in results]
            rerank_scores = self.reranker.rerank(query, texts, batch_size=rerank_batch_size)

            if rerank_scores is not None:
                # Add rerank scores to results
                for i, result in enumerate(results):
                    result["score_rerank"] = float(rerank_scores[i])

                # Sort by rerank score (primary), then by original score (secondary)
                score_key = "score_hybrid" if search_mode == "hybrid" else "score_vec"
                results.sort(key=lambda r: (r.get("score_rerank", 0), r.get(score_key, 0)), reverse=True)
            else:
                # Reranking failed, keep original order
                score_key = "score_hybrid" if search_mode == "hybrid" else "score_vec"
                results.sort(key=lambda r: r.get(score_key, 0), reverse=True)
        else:
            # No reranking, sort by original scores
            if search_mode == "hybrid":
                results.sort(key=lambda r: r.get("score_hybrid", 0), reverse=True)
            elif search_mode == "bm25":
                results.sort(key=lambda r: r.get("score_bm25", 0), reverse=True)
            else:
                results.sort(key=lambda r: r.get("score_vec", 0), reverse=True)

        # Return top-k results
        return results[:top_k]
