# part_I_system_components_fastapi.py
"""
FastAPI version of the Medical SME System.
Migrated from Flask while preserving all functionality and contracts.
"""
import os
import json
import logging
from typing import Optional, Dict, Any, List
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import JSONResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel, Field

from part_H_tools import (
    export_qa_pdf, export_qa_docx,
    export_quiz_pdf, export_quiz_docx, export_quiz_pptx
)
from helpers import sanitize_for_injection
from part_A_collection import collect_and_organize_documents
from part_B_preprocessing import run_batch
from part_DEF_agent_llm_capabilities import SearchDocsTool
from custom_agent import ConversationMemory
from part_G_RAG import load_elasticsearch_config, ElasticsearchIndex
from llm_config import get_config_manager, ALLOWED_MODELS

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ============================================================================
# Pydantic Models for Request Validation
# ============================================================================

class QARequest(BaseModel):
    question: str

class QuizRequest(BaseModel):
    topic: str
    n: int = Field(default=5, ge=1, le=20)

class ChatRequest(BaseModel):
    query: str
    session_id: str = "default_session"

class ClearChatRequest(BaseModel):
    session_id: str = "default_session"

class ExportQARequest(BaseModel):
    question: str
    answer: str
    sources: List[Dict[str, Any]] = []

class ExportQuizRequest(BaseModel):
    topic: str
    items: List[Dict[str, Any]]
    sources: List[Dict[str, Any]] = []

class IngestRequest(BaseModel):
    input_dir: Optional[str] = None
    artifacts_dir: Optional[str] = None
    granularities: Optional[str] = None
    overlap_tokens: Optional[int] = None
    write_metadata: bool = True

class ESCreateIndexRequest(BaseModel):
    delete_existing: bool = False
    dims: Optional[int] = None

class LLMConfigRequest(BaseModel):
    model_name: Optional[str] = None
    temperature: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    max_new_tokens: Optional[int] = Field(default=None, ge=64, le=2048)

# ============================================================================
# Utility Functions (preserved from Flask version)
# ============================================================================

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
    write_metadata=True,
    es_index=None
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
            es_index.create_index(delete_if_exists=False)
    except Exception as e:
        # Keep ingestion working even if ES is down
        print(f"[batch_ingestion_pipeline] Elasticsearch unavailable, continuing without ES: {e}")
        es_index = None

    run_batch(
        input_dir=input_dir,
        artifacts_dir=artifacts_dir,
        granularities=list(granularities),
        overlap_tokens=int(overlap_tokens),
        es_index=es_index
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


# ============================================================================
# Global State (preserved from Flask version)
# ============================================================================

INGESTION_STATUS = None  # Will be set in startup event
CONVERSATION_MEMORIES: Dict[str, ConversationMemory] = {}

# Search tool and config manager
search_tool = None
config_manager = get_config_manager()

# ============================================================================
# FastAPI Application
# ============================================================================

app = FastAPI(
    title="Medical SME System",
    description="Subject Matter Expert for Neuroanatomy with RAG, Quiz Generation, and Document Export",
    version="2.0.0"
)

# Mount static files
app.mount("/static", StaticFiles(directory="static"), name="static")

# Setup templates
templates = Jinja2Templates(directory="templates")


# ============================================================================
# Startup Event
# ============================================================================

@app.on_event("startup")
async def startup_event():
    """Initialize system on startup."""
    global INGESTION_STATUS, search_tool

    logger.info("Starting Medical SME System...")

    # Run ingestion
    INGESTION_STATUS = run_ingestion()
    logger.info(f"Ingestion status: {INGESTION_STATUS}")

    # Initialize search tool
    search_mode = os.environ.get("SEARCH_MODE", "hybrid")
    use_es = env_bool("USE_ELASTICSEARCH", True)
    search_tool = SearchDocsTool(
        k=8, candidates=120, device=None,
        search_mode=search_mode, use_elasticsearch=use_es
    )
    logger.info(f"Search tool initialized (mode: {search_mode}, ES: {use_es})")

    # Initialize LLM components with default config
    config_manager.get_or_create_agent(search_tool)
    logger.info("LLM components initialized")


# ============================================================================
# Frontend Routes
# ============================================================================

@app.get("/")
async def root():
    """Redirect to chat page."""
    return RedirectResponse(url="/chat", status_code=302)


@app.get("/chat")
async def chat_page(request: Request):
    """Render chat interface."""
    return templates.TemplateResponse("chat.html", {"request": request})


@app.get("/tools")
async def tools_page(request: Request):
    """Render QA/Quiz tools interface."""
    return templates.TemplateResponse("tools.html", {"request": request})


@app.get("/health")
async def health():
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

    return status


# ============================================================================
# LangChain Direct Endpoints (QA & Quiz)
# ============================================================================

@app.post("/lc/qa")
async def lc_qa(request: QARequest):
    """Direct QA using LangChain chain."""
    question = sanitize_for_injection(request.question)

    # Get current QA chain
    _, _, qa_chain, _ = config_manager.get_or_create_agent(search_tool)

    if qa_chain is None:
        raise HTTPException(status_code=500, detail="LangChain unavailable")

    out = qa_chain.invoke({"question": question})
    return out


@app.post("/lc/quiz")
async def lc_quiz(request: QuizRequest):
    """Direct quiz generation using LangChain chain."""
    topic = request.topic
    n = request.n

    # Get current LLM agent to build quiz chain
    llm_agent, llm, _, _ = config_manager.get_or_create_agent(search_tool)

    if not search_tool:
        raise HTTPException(status_code=500, detail="Search tool unavailable")

    quiz_chain = llm_agent.build_lc_quiz_chain(llm, search_tool, n_questions=n)
    out = quiz_chain.invoke({"topic": topic})
    return out


# ============================================================================
# Chat Endpoint (Agentic with Memory)
# ============================================================================

@app.post("/chat")
async def chat(request: ChatRequest):
    """
    Main conversational agent endpoint (Custom Executor Version).
    Manages chat history via ConversationMemory.
    ENHANCED: Now includes intermediate_steps in response.
    """
    try:
        query = sanitize_for_injection(request.query)
        session_id = request.session_id

        if not query:
            raise HTTPException(status_code=400, detail="Query cannot be empty")

        # Get or create conversation memory for this session
        if session_id not in CONVERSATION_MEMORIES:
            CONVERSATION_MEMORIES[session_id] = ConversationMemory(max_history=10)
        memory = CONVERSATION_MEMORIES[session_id]

        logger.info(f"Processing query for session {session_id}: {query}")

        # Get current agent executor
        llm_agent, _, _, agent_executor = config_manager.get_or_create_agent(search_tool)

        # Invoke the agent with memory
        result = llm_agent.invoke_agent(agent_executor, query, memory)

        agent_output = result['output']  # This is the default ("Here is a quiz...")

        intermediate_steps = result.get('intermediate_steps', [])
        if intermediate_steps:
            last_step = intermediate_steps[-1]
            last_action = last_step.get("action")

            # Check if the last action was one that should return its data
            if last_action in ["generate_quiz_on_neuroanatomy_topic", "answer_question_about_neuroanatomy"]:
                observation = last_step.get("observation")

                # Check if it's the data object, not an error string
                if isinstance(observation, dict):
                    agent_output = observation  # This is now {"items": [...]}
                else:
                    agent_output = str(observation)  # It's an error

            # Handle export/email tools, which just return a string path/confirmation
            elif last_action in ["export_document", "send_email"]:
                agent_output = str(last_step.get("observation"))

        # Serialize intermediate steps for UI
        serialized_steps = []
        for step in intermediate_steps:
            serialized_steps.append({
                "thought": step.get("thought", ""),
                "action": step.get("action", ""),
                "action_input": step.get("action_input", {}),
                "observation": str(step.get("observation", ""))[:500]  # Limit length
            })

        # Update conversation memory
        # We must save a string to memory, so we convert back
        memory_output = json.dumps(agent_output) if isinstance(agent_output, dict) else str(agent_output)
        memory.add_user_message(query)
        memory.add_assistant_message(memory_output)

        logger.info(f"Agent response for session {session_id}: {memory_output[:100]}...")

        # Return response with intermediate steps
        return {
            "response": agent_output,
            "session_id": session_id,
            "tools_used": len(result.get('intermediate_steps', [])),
            "intermediate_steps": serialized_steps
        }

    except Exception as e:
        logger.error(f"Chat endpoint failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/chat/clear")
async def clear_chat(request: ClearChatRequest):
    """Clear conversation history for a session."""
    try:
        session_id = request.session_id

        if session_id in CONVERSATION_MEMORIES:
            CONVERSATION_MEMORIES[session_id].clear()
            logger.info(f"Cleared conversation history for session {session_id}")
            return {"ok": True, "message": "History cleared"}
        else:
            return {"ok": True, "message": "No history to clear"}
    except Exception as e:
        logger.error(f"Clear chat failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/chat/history/{session_id}")
async def get_chat_history(session_id: str):
    """Get conversation history for a session."""
    try:
        if session_id in CONVERSATION_MEMORIES:
            history = CONVERSATION_MEMORIES[session_id].get_history()
            return {
                "session_id": session_id,
                "history": history,
                "message_count": len(history)
            }
        else:
            return {
                "session_id": session_id,
                "history": [],
                "message_count": 0
            }
    except Exception as e:
        logger.error(f"Get history failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================================
# Export Endpoints
# ============================================================================

@app.post("/export/qa")
async def export_qa_route(request: ExportQARequest):
    """Export QA to PDF/DOCX."""
    question = request.question
    answer = request.answer
    sources = request.sources

    os.makedirs("./exports", exist_ok=True)
    export_qa_pdf(question, answer, sources, "./exports/qa.pdf")
    export_qa_docx(question, answer, sources, "./exports/qa.docx")

    return {"ok": True, "paths": ["./exports/qa.pdf", "./exports/qa.docx"]}


@app.post("/export/quiz")
async def export_quiz_route(request: ExportQuizRequest):
    """Export quiz to PDF/DOCX/PPTX."""
    topic = request.topic
    items = request.items
    sources = request.sources

    os.makedirs("./exports", exist_ok=True)
    export_quiz_pdf(topic, items, sources, "./exports/quiz.pdf")
    export_quiz_docx(topic, items, sources, "./exports/quiz.docx")
    export_quiz_pptx(topic, items, sources, "./exports/quiz.pptx")

    return {"ok": True, "paths": ["./exports/quiz.pdf", "./exports/quiz.docx", "./exports/quiz.pptx"]}


# ============================================================================
# Admin Endpoints
# ============================================================================

@app.post("/admin/ingest")
async def admin_ingest(request: IngestRequest):
    """Trigger manual ingestion pipeline."""
    input_dir = request.input_dir or os.environ.get("SME_INPUT_DIR", "./data")
    artifacts_dir = request.artifacts_dir or os.environ.get("SME_ARTIFACTS_DIR", "./artifacts")
    granularities = parse_int_list(
        request.granularities or os.environ.get("SME_GRANULARITIES"),
        (2048, 512, 128)
    )
    overlap_tokens = request.overlap_tokens or int(os.environ.get("SME_OVERLAP_TOKENS", "64"))
    write_meta = request.write_metadata

    out = batch_ingestion_pipeline(
        input_dir=input_dir,
        artifacts_dir=artifacts_dir,
        granularities=tuple(granularities),
        overlap_tokens=overlap_tokens,
        write_metadata=write_meta,
    )
    return out


@app.get("/admin/search-backend")
async def admin_search_backend():
    """Get current search backend configuration."""
    try:
        config = load_elasticsearch_config()
        if config and config.get("enabled", True):
            try:
                es_index = ElasticsearchIndex(config=config)
                doc_count = es_index.get_document_count()
                return {
                    "backend": "elasticsearch",
                    "enabled": True,
                    "host": config.get("host"),
                    "port": config.get("port"),
                    "index_name": config.get("index_name"),
                    "document_count": doc_count,
                    "status": "connected"
                }
            except Exception as e:
                return {
                    "backend": "elasticsearch",
                    "enabled": True,
                    "status": "error",
                    "error": str(e)
                }
        else:
            return {
                "backend": "faiss",
                "reason": "elasticsearch_disabled_in_config"
            }
    except Exception:
        return {
            "backend": "faiss",
            "reason": "elasticsearch_config_not_found"
        }


@app.post("/admin/es/create-index")
async def admin_es_create_index(request: ESCreateIndexRequest):
    """Create Elasticsearch index with proper mapping."""
    try:
        config = load_elasticsearch_config()
        if not config or not config.get("enabled", True):
            raise HTTPException(status_code=400, detail="Elasticsearch not enabled")

        delete_existing = request.delete_existing

        # Default: use config["vector_dims"], allow explicit override in payload
        default_dims = int(config.get("vector_dims", 768))
        dims = request.dims or default_dims

        es_index = ElasticsearchIndex(config=config)
        es_index.create_index(dims=dims, delete_if_exists=delete_existing)

        return {
            "ok": True,
            "index_name": es_index.index_name,
            "dims": dims,
            "deleted_existing": delete_existing
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/admin/es/reindex")
async def admin_es_reindex():
    """Reindex all chunks from JSONL artifacts to Elasticsearch."""
    try:
        config = load_elasticsearch_config()
        if not config or not config.get("enabled", True):
            raise HTTPException(status_code=400, detail="Elasticsearch not enabled")

        import glob
        from part_G_RAG import _json_lines

        es_index = ElasticsearchIndex(config=config)
        default_dims = int(config.get("vector_dims", 768))
        es_index.create_index(dims=default_dims, delete_if_exists=False)

        # Find all node files
        node_files = glob.glob("./artifacts/graph/*__nodes.jsonl")
        if not node_files:
            raise HTTPException(status_code=404, detail="No graph node files found")

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

        return {
            "ok": True,
            "files_processed": len(node_files),
            "chunks_indexed": total_indexed,
            "chunks_failed": total_failed
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/admin/llm-config")
async def admin_llm_config(request: LLMConfigRequest):
    """
    Update LLM configuration dynamically.
    Changes take effect for subsequent requests without server restart.
    """
    try:
        # Update configuration
        success, error = config_manager.update_config(
            model_name=request.model_name,
            temperature=request.temperature,
            max_new_tokens=request.max_new_tokens
        )

        if not success:
            raise HTTPException(status_code=400, detail=error)

        # Get updated config
        current_config = config_manager.get_config()

        return {
            "ok": True,
            "message": "LLM configuration updated successfully",
            "config": {
                "model_name": current_config.model_name,
                "temperature": current_config.temperature,
                "max_new_tokens": current_config.max_new_tokens
            },
            "allowed_models": list(ALLOWED_MODELS)
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"LLM config update failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/admin/llm-config")
async def get_llm_config():
    """Get current LLM configuration."""
    current_config = config_manager.get_config()
    return {
        "model_name": current_config.model_name,
        "temperature": current_config.temperature,
        "max_new_tokens": current_config.max_new_tokens,
        "allowed_models": list(ALLOWED_MODELS)
    }


# ============================================================================
# Main Entry Point
# ============================================================================

if __name__ == "__main__":
    import uvicorn

    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(
        "part_I_system_components_fastapi:app",
        host="0.0.0.0",
        port=port,
        reload=False,
        log_level="info"
    )
