
from functools import partial
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline, BitsAndBytesConfig
from langchain_community.llms.huggingface_pipeline import HuggingFacePipeline
from operator import itemgetter
import random
import torch
import os
import json
import random
from typing import Optional, Any
from pydantic import PrivateAttr
from langchain_core.prompts import PromptTemplate
from langchain_core.tools import BaseTool
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda, RunnableParallel
from part_G_RAG import (
    EmbeddingIndex, ElasticsearchIndex, QueryEncoder, BGEReranker,
    SearchPipeline, load_elasticsearch_config
)
import logging

logger = logging.getLogger(__name__)

bnb_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_type="bfloat16")


def create_search_index(config_path=None, use_elasticsearch=True):
    if use_elasticsearch:
        try:
            config = load_elasticsearch_config(config_path)
            if config and config.get("enabled", True):
                es_index = ElasticsearchIndex(config=config)

                # Ensure index exists
                es_index.create_index(dims=768, delete_if_exists=False)
                logger.info("Using Elasticsearch backend for search")
                return es_index
        except Exception as e:
            logger.warning(f"Elasticsearch initialization failed: {e}")
            logger.info("Falling back to FAISS backend")

    # Fallback to FAISS
    logger.info("Using FAISS backend for search")
    return EmbeddingIndex.from_files()

class SearchDocsTool(BaseTool):
    name: str = "search_docs"
    description: str = "Semantic search over textbook chunks of the human nervous system corpus."
    k: int = 8
    candidates: int = 120
    device: Optional[str] = None
    search_mode: str = "hybrid"  # "vector", "bm25", or "hybrid"

    _index: Any = PrivateAttr(default=None)
    _encoder: Any = PrivateAttr(default=None)
    _reranker: Any = PrivateAttr(default=None)
    _pipe: Any = PrivateAttr(default=None)

    def __init__(self, k=8, candidates=120, device=None, search_mode="hybrid",
                 config_path=None, use_elasticsearch=True):
        """
        Initialize SearchDocsTool with either Elasticsearch or FAISS backend.

        Args:
            k: Number of results to return
            candidates: Number of candidates for reranking
            device: Device for models (cuda/cpu)
            search_mode: "vector", "bm25", or "hybrid" (default: "hybrid")
            config_path: Path to ES config file (optional)
            use_elasticsearch: Try to use ES if True (default: True)
        """
        super(SearchDocsTool, self).__init__()
        self.search_mode = search_mode

        # Create index (ES or FAISS fallback)
        self._index = create_search_index(config_path, use_elasticsearch)

        # Create encoder and reranker
        self._encoder = QueryEncoder(device=device)
        self._reranker = BGEReranker(device=device)

        # Create search pipeline
        self._pipe = SearchPipeline(self._index, self._encoder, self._reranker)

    def _run(self, query):
        hits = self._pipe.search(
            query,
            top_k=self.k,
            candidates=self.candidates,
            use_reranker=True,
            search_mode=self.search_mode
        )
        out = []
        for h in hits:
            out.append({
                "text": h.get("text") or "",
                "book": h.get("source_basename"),
                "chunk_id": h.get("chunk_id"),
                "granularity": h.get("granularity_tokens"),
                "position": h.get("position"),
                "score_vec": h.get("score_vec"),
                "score_bm25": h.get("score_bm25"),
                "score_hybrid": h.get("score_hybrid"),
                "score_rerank": h.get("score_rerank"),
            })
        return out

def parse_quiz_json(text):
    s = text.strip()
    s = s[s.find('{'):] if '{' in s else s
    s = s[:s.rfind('}')+1] if '}' in s else s
    try:
        obj = json.loads(s)
        q = obj.get("question")
        opts = obj.get("options", {})
        corr = obj.get("correct")
        if not q or not isinstance(opts, dict) or set(opts.keys()) != {"A","B","C","D"} or corr not in opts:
            return None
        return {"question": q, "options": opts, "correct": corr}
    except Exception:
        return None



def load_medgemma_llm_lc(max_new_tokens=512, temperature=0.2, bnb_config=bnb_config, model_name="google/medgemma-4b-it"):
    tok = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype="auto", device_map="auto", trust_remote_code=True, quantization_config=bnb_config
    )
    gen = pipeline(
        "text-generation",
        model=model,
        tokenizer=tok,
        max_new_tokens=max_new_tokens,
        do_sample=temperature > 0,
        temperature=temperature,
        return_full_text=False,
    )
    return HuggingFacePipeline(pipeline=gen)

def lc_templates():
    qa = PromptTemplate.from_template(
        "Answer ONLY from the provided context. If missing, say 'Insufficient context.'\n\nContext:\n{context}\n\nQuestion: {question}\nAnswer:"
    )
    quiz = PromptTemplate.from_template(
        "Create ONE MCQ (A–D) from the context. One correct option only. Return JSON with question, options (A..D), correct.\n\nContext:\n{context}\n\nJSON:"
    )
    return qa, quiz

def add_context(docs, k=3):
    chunks = []
    for d in docs[:k]:
        t = (d.get("text") or "")
        chunks.append(t)
    return "\n\n---\n\n".join(chunks)

def build_lc_qa_chain(llm, search_tool):
    fan = RunnableParallel(
        question=itemgetter("question"),
        docs=RunnableLambda(lambda x: search_tool.run(x["question"])),
    )
   
    qa_t, _ = lc_templates()
    compose = RunnableLambda(lambda x: {
        "question": x["question"],
        "context": add_context(x["docs"]),
        "docs": x["docs"]
    })
    chain = fan | compose | {"answer": (qa_t | llm | StrOutputParser()), "sources": (RunnableLambda(lambda y: y["docs"]))}
    return chain


def generate_single_mcq_from_context(context_text, llm, quiz_prompt):
    raw = (quiz_prompt | llm | StrOutputParser()).invoke({"context": context_text})
    return parse_quiz_json(raw)

def build_quiz_items_from_topic(topic, search_tool, llm, quiz_prompt, n_questions=5):
    docs = search_tool.run(topic)
    shuffled = list(docs)
    random.shuffle(shuffled)
    items = []
    for d in shuffled:
        if len(items) >= n_questions:
            break
        context_text = (d.get("text") or "")
        parsed = generate_single_mcq_from_context(context_text, llm, quiz_prompt)
        if parsed:
            parsed["source"] = {
                "book": d.get("book"),
                "chunk_id": d.get("chunk_id"),
                "granularity": d.get("granularity"),
                "position": d.get("position"),
            }
            items.append(parsed)
    return {"items": items}

def generate_quiz_payload(payload, search_tool, llm, quiz_prompt, n_questions):
    topic = payload["topic"]
    return build_quiz_items_from_topic(topic, search_tool, llm, quiz_prompt, n_questions=n_questions)

def build_lc_quiz_chain(llm, search_tool, n_questions=5):
    quiz_prompt = lc_templates()[1]
    bound = partial(
        generate_quiz_payload,
        search_tool=search_tool,
        llm=llm,
        quiz_prompt=quiz_prompt,
        n_questions=n_questions,
    )
    return RunnableLambda(bound)




