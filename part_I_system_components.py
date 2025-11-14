# part_I_system_components.py
import os
import time
import json
from contextlib import contextmanager
from flask import Flask, request, jsonify
from part_H_tools import (
    export_qa_pdf, export_qa_docx,
    export_quiz_pdf, export_quiz_docx, export_quiz_pptx
)

from helpers import sanitize_for_injection
from part_A_collection import collect_and_organize_documents
from part_B_preprocessing import run_batch
from part_DEF_agent_llm_capabilities import (
    SearchDocsTool,
    load_medgemma_llm_lc,
    build_lc_qa_chain,
    build_lc_quiz_chain,
    create_search_index,
)
from part_G_RAG import load_elasticsearch_config, ElasticsearchIndex

def env_bool(name, default=False):
    v = os.environ.get(name)
    if v is None:
        return default
    return str(v).strip().lower() in {"1", "true", "yes", "y", "on"}

def parse_int_list(csv, default):
    if not csv:
        return list(default)
    try:
        return [int(x.strip()) for x in str(csv).split(",") if x.strip()]
    except Exception:
        return list(default)

def batch_ingestion_pipeline(
    input_dir="./data",
    artifacts_dir="./artifacts",
    granularities=(2048, 512, 128),
    overlap_tokens=64,
    write_metadata=True
):
    os.makedirs(artifacts_dir, exist_ok=True)
    # Writing Metadata from Part A
    if write_metadata:
        corpus, metadata = collect_and_organize_documents(input_dir)
        os.makedirs("./metadata", exist_ok=True)
        with open("./metadata/file_metadata.json", "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2)

    # Writing Elasticsearch Indexing
    es_index = None
    try:
        config = load_elasticsearch_config()
        if config and config.get("enabled", True):
            es_index = ElasticsearchIndex(config=config)
            # Do NOT delete existing index by default.
            es_index.create_index(dims=768, delete_if_exists=False)
    except Exception as e:
        # Keep ingestion working even if ES is down
        print(f"[batch_ingestion_pipeline] Elasticsearch unavailable, continuing without ES: {e}")
        es_index = None
        
    run_batch(
        input_dir=input_dir,
        artifacts_dir=artifacts_dir,
        granularities=list(granularities),
        overlap_tokens=int(overlap_tokens),
        
    )

    return {"started": True, "artifacts_dir": artifacts_dir}


def get_ingestion_flag():
    auto = env_bool("SME_INGEST_ON_START", True)
    if not auto:
        return False
    import glob
    for pat in (
        "./artifacts/embeddings/*__emb.jsonl",
        "./artifacts/graph/*__nodes.jsonl",
        "./artifacts/chunks/*.jsonl",
    ):
        if glob.glob(pat, recursive=True):
            return False
    return True


def run_ingestion():
    if not get_ingestion_flag():
        return {"skipped": True}

    granularities = parse_int_list(os.environ.get("SME_GRANULARITIES"), (2048, 512, 128))
    overlap_tokens = int(os.environ.get("SME_OVERLAP_TOKENS", "64"))
    write_meta = not env_bool("SME_SKIP_METADATA", False)

    return batch_ingestion_pipeline(
        granularities=tuple(granularities),
        overlap_tokens=overlap_tokens,
        write_metadata=write_meta,
    )


INGESTION_STATUS = run_ingestion()

def create_app():
    app = Flask(__name__)
    @app.get("/health")
    def health():
        """Health check with search backend status."""
        status = {"ok": True, "ingest": INGESTION_STATUS}

        # Check search backend
        try:
            config = load_elasticsearch_config()
            if config and config.get("enabled", True):
                try:
                    es_index = ElasticsearchIndex(config=config)
                    status["search_backend"] = "elasticsearch"
                    status["es_connected"] = True
                    status["es_document_count"] = es_index.get_document_count()
                except Exception:
                    status["search_backend"] = "faiss"
                    status["es_connected"] = False
                    status["es_error"] = "connection_failed"
            else:
                status["search_backend"] = "faiss"
                status["es_enabled"] = False
        except Exception:
            status["search_backend"] = "faiss"
            status["es_config_found"] = False

        return jsonify(status)
    @app.post("/export/qa")
    def export_qa_route():
        data = request.get_json(force=True) or {}
        answer = data.get("answer", "")
        sources = data.get("sources", [])
        os.makedirs("./exports", exist_ok=True)
        export_qa_pdf(answer, sources, "./exports/qa.pdf")
        export_qa_docx(answer, sources, "./exports/qa.docx")
        return jsonify({"ok": True, "paths": ["./exports/qa.pdf", "./exports/qa.docx"]})

    @app.post("/export/quiz")
    def export_quiz_route():
        data = request.get_json(force=True) or {}
        items = data.get("items", [])
        os.makedirs("./exports", exist_ok=True)
        export_quiz_pdf(items, "./exports/quiz.pdf")
        export_quiz_docx(items, "./exports/quiz.docx")
        export_quiz_pptx(items, "./exports/quiz.pptx")
        return jsonify({"ok": True, "paths": ["./exports/quiz.pdf", "./exports/quiz.docx", "./exports/quiz.pptx"]})

    @app.post("/admin/ingest")
    def admin_ingest():
        payload = request.get_json(silent=True) or {}
        input_dir = payload.get("input_dir", os.environ.get("SME_INPUT_DIR", "./data"))
        artifacts_dir = payload.get("artifacts_dir", os.environ.get("SME_ARTIFACTS_DIR", "./artifacts"))
        granularities = parse_int_list(payload.get("granularities") or os.environ.get("SME_GRANULARITIES"), (2048, 512, 128))
        overlap_tokens = int(payload.get("overlap_tokens") or os.environ.get("SME_OVERLAP_TOKENS", "64"))
        write_meta = bool(payload.get("write_metadata", True))
        out = batch_ingestion_pipeline(
            input_dir=input_dir,
            artifacts_dir=artifacts_dir,
            granularities=tuple(granularities),
            overlap_tokens=overlap_tokens,
            write_metadata=write_meta,
        )
        return jsonify(out)

    @app.get("/admin/search-backend")
    def admin_search_backend():
        """Get current search backend configuration."""
        try:
            config = load_elasticsearch_config()
            if config and config.get("enabled", True):
                try:
                    es_index = ElasticsearchIndex(config=config)
                    doc_count = es_index.get_document_count()
                    return jsonify({
                        "backend": "elasticsearch",
                        "enabled": True,
                        "host": config.get("host"),
                        "port": config.get("port"),
                        "index_name": config.get("index_name"),
                        "document_count": doc_count,
                        "status": "connected"
                    })
                except Exception as e:
                    return jsonify({
                        "backend": "elasticsearch",
                        "enabled": True,
                        "status": "error",
                        "error": str(e)
                    })
            else:
                return jsonify({
                    "backend": "faiss",
                    "reason": "elasticsearch_disabled_in_config"
                })
        except Exception:
            return jsonify({
                "backend": "faiss",
                "reason": "elasticsearch_config_not_found"
            })

    @app.post("/admin/es/create-index")
    def admin_es_create_index():
        """Create Elasticsearch index with proper mapping."""
        try:
            config = load_elasticsearch_config()
            if not config or not config.get("enabled", True):
                return jsonify({"error": "Elasticsearch not enabled"}), 400

            payload = request.get_json(silent=True) or {}
            delete_existing = payload.get("delete_existing", False)
            dims = int(payload.get("dims", 768))

            es_index = ElasticsearchIndex(config=config)
            es_index.create_index(dims=dims, delete_if_exists=delete_existing)

            return jsonify({
                "ok": True,
                "index_name": es_index.index_name,
                "dims": dims,
                "deleted_existing": delete_existing
            })
        except Exception as e:
            return jsonify({"error": str(e)}), 500

    @app.post("/admin/es/reindex")
    def admin_es_reindex():
        """Reindex all chunks from JSONL artifacts to Elasticsearch."""
        try:
            config = load_elasticsearch_config()
            if not config or not config.get("enabled", True):
                return jsonify({"error": "Elasticsearch not enabled"}), 400

            import glob
            from part_G_RAG import _json_lines

            es_index = ElasticsearchIndex(config=config)
            es_index.create_index(dims=768, delete_if_exists=False)

            # Find all node files
            node_files = glob.glob("./artifacts/graph/*__nodes.jsonl")
            if not node_files:
                return jsonify({"error": "No graph node files found"}), 404

            total_indexed = 0
            total_failed = 0

            for node_file in node_files:
                chunks = []
                for node in _json_lines(node_file):
                    if node.get("node_type") == "chunk":
                        chunk_data = {
                            "chunk_id": node.get("node_id"),
                            "doc_id": node.get("doc_id"),
                            "text": node.get("text", ""),
                            "embedding": node.get("embedding"),
                            "source_basename": node.get("source_basename"),
                            "granularity_tokens": node.get("granularity_tokens"),
                            "position": node.get("position"),
                            "created_at": node.get("created_at"),
                            "parent_doc_hash": node.get("parent_doc_hash"),
                            "n_tokens": node.get("n_tokens"),
                            "source_path": node.get("source_path")
                        }
                        chunks.append(chunk_data)

                if chunks:
                    success, failed = es_index.index_chunks_bulk(chunks)
                    total_indexed += success
                    total_failed += failed

            return jsonify({
                "ok": True,
                "files_processed": len(node_files),
                "chunks_indexed": total_indexed,
                "chunks_failed": total_failed
            })
        except Exception as e:
            return jsonify({"error": str(e)}), 500

    # Initialize search tool with ES support
    search_mode = os.environ.get("SEARCH_MODE", "hybrid")
    use_es = env_bool("USE_ELASTICSEARCH", True)
    search_tool = SearchDocsTool(k=8, candidates=120, device=None,
                                  search_mode=search_mode, use_elasticsearch=use_es)
    llm = load_medgemma_llm_lc()
    qa_chain = build_lc_qa_chain(llm, search_tool) if search_tool else None

    @app.post("/lc/qa")
    def lc_qa():
        data = request.get_json(force=True) or {}
        question = sanitize_for_injection(data.get("question", ""))
        out = qa_chain.invoke({"question": question}) if qa_chain else {"error": "LangChain unavailable"}
        return jsonify(out)

    @app.post("/lc/quiz")
    def lc_quiz():
        data = request.get_json(force=True) or {}
        topic = data.get("topic", "")
        n = int(data.get("n", 5))
        quiz_chain = build_lc_quiz_chain(llm, search_tool, n_questions=n) if search_tool else None
        out = quiz_chain.invoke({"topic": topic}) if quiz_chain else {"error": "LangChain unavailable"}
        return jsonify(out)

    return app


app = create_app()

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 8000)), debug=False, use_reloader=False)
