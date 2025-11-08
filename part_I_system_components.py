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

from part_A_collection import collect_and_organize_documents
from part_B_preprocessing import run_batch
from part_DEF_agent_llm_capabilities import (
    SearchDocsTool,
    load_medgemma_llm_lc,
    build_lc_qa_chain,
    build_lc_quiz_chain,
)

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
    if write_metadata:
        corpus, metadata = collect_and_organize_documents(input_dir)
        os.makedirs("./metadata", exist_ok=True)
        with open("./metadata/file_metadata.json", "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2)
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
        return jsonify({"ok": True, "ingest": INGESTION_STATUS})
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
    search_tool = SearchDocsTool(k=8, candidates=120, device=None)
    llm = load_medgemma_llm_lc()
    qa_chain = build_lc_qa_chain(llm, search_tool) if search_tool else None

    @app.post("/lc/qa")
    def lc_qa():
        data = request.get_json(force=True) or {}
        question = data.get("question", "")
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
