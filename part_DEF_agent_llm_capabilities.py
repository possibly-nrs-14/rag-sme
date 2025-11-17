# part_DEF_agent_llm_capabilities.py
from functools import partial
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline, BitsAndBytesConfig
from langchain_huggingface import HuggingFacePipeline
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
                es_index.create_index(dims=None, delete_if_exists=False)
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


class LLMAgent():
    def __init__(self, model_name='google/medgemma-4b-it', max_new_tokens=256, temperature=0.05):
        self.model_name = model_name
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
    def load_llm_lc(self):
        tok = AutoTokenizer.from_pretrained(self.model_name, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            self.model_name, torch_dtype="auto", device_map="auto", trust_remote_code=True, quantization_config=bnb_config
        )
        gen = pipeline(
            "text-generation",
            model=model,
            tokenizer=tok,
            max_new_tokens=self.max_new_tokens,
            do_sample=self.temperature > 0,
            temperature=self.temperature,
            return_full_text=False,
        )
        return HuggingFacePipeline(pipeline=gen)

    def lc_templates(self):
        qa, quiz = None, None
        if self.model_name == 'google/medgemma-4b-it':
            qa = PromptTemplate.from_template(
                "You are a neuroanatomy assistant.\n"
                "Use ONLY the information in the Context to answer the Question.\n"
                "You must output exactly ONE of the following:\n"
                "1) A single, self-contained answer in at most 3 sentences of plain English, "
                "IF the Context clearly contains the answer.\n"
                "2) Exactly the phrase: Insufficient context. (nothing else), "
                "IF the Context does NOT contain the answer.\n"
                "You MUST NOT output both an answer AND 'Insufficient context.'\n"
                "Do NOT write code, pseudo-code, or functions.\n"
                "Do NOT include backticks or markdown fences in your reply.\n\n"
                "Context:\n{context}\n\n"
                "Question: {question}\n"
                "Answer:"
            )
            quiz = PromptTemplate.from_template(
                "Create ONE MCQ (A–D) from the context. One correct option only. Return JSON with question, options (A..D), correct.\n\nContext:\n{context}\n\nJSON:"
            )
        elif self.model_name == 'Intelligent-Internet/II-Medical-8B':
            qa = PromptTemplate.from_template(
                "<|system|>\n"
                "You are an expert neuroanatomy assistant. Your task is to answer the user's question based *only* on the provided context.\n"
                "Follow these steps:\n"
                "1.  Carefully read the Question and the Context.\n"
                "2.  Reason step-by-step to determine if the Context contains the information to answer the Question.\n"
                "3.  If the answer is in the context, formulate a concise, one-paragraph answer.\n"
                "4.  If the answer is NOT in the context, your final answer MUST be exactly: Insufficient context.\n"
                "5.  Provide your reasoning and final answer in the specified format.\n\n"
                "<|user|>\n"
                "**Context:**\n"
                "{context}\n\n"
                "**Question:**\n"
                "{question}\n\n"
                "<|assistant|>\n"
                
                "[Your final answer. This should be a concise paragraph OR the exact phrase 'Insufficient context.']"
            )
            quiz = PromptTemplate.from_template(
                "Create ONE MCQ (A–D) from the context. One correct option only. Return JSON with question, options (A..D), correct.\n\nContext:\n{context}\n\nJSON:"
            )
        elif self.model_name == 'microsoft/MediPhi':
            qa = PromptTemplate.from_template(
                    "<|system|>\n"
                    "You are an expert clinical QA assistant.\n"
                    "You must answer the user's question based *only* on the provided context.\n"
                    "- If the context contains the answer, provide a concise, single-paragraph answer.\n"
                    "- If the context does NOT contain the answer, you MUST respond with *only* the exact phrase: Insufficient context.\n"
                    "<|end|>\n"
                    "<|user|>\n"
                    "Context:\n"
                    "{context}\n\n"
                    "Question: {question}\n"
                    "<|end|>\n"
                    "<|assistant|>"
                )
            
            quiz = PromptTemplate.from_template(
                "Create ONE MCQ (A–D) from the context. One correct option only. Return JSON with question, options (A..D), correct.\n\nContext:\n{context}\n\nJSON:"
            )
            # quiz = PromptTemplate.from_template(
            #     "Create ONE MCQ (A–D) from the context. Exactly one correct option.\n"
            #     "Return ONLY a single JSON object with keys:\n"
            #     "- question (string)\n"
            #     "- options (object with keys 'A','B','C','D')\n"
            #     "- correct (one of 'A','B','C','D')\n"
            #     "Do NOT include any explanation, markdown, or text outside the JSON.\n\n"
            #     "Context:\n{context}\n\nJSON:"
            # )

        return qa, quiz

    def add_context(self, docs, k=3):
        chunks = []
        for d in docs[:k]:
            t = (d.get("text") or "")
            chunks.append(t)
        return "\n\n---\n\n".join(chunks)
    
    def dedupe_lines(self, text):
        seen = set()
        out_lines = []
        for line in text.splitlines():
            l = line.strip()
            if not l:
                continue
            if l in seen:
                continue
            seen.add(l)
            out_lines.append(line)
        return "\n".join(out_lines)
    
    def build_lc_qa_chain(self, llm, search_tool):
        fan = RunnableParallel(
            question=itemgetter("question"),
            docs=RunnableLambda(lambda x: search_tool.run(x["question"])),
        )
    
        qa_t, _ = self.lc_templates()

        compose = RunnableLambda(lambda x: {
            "question": x["question"],
            "context": self.add_context(x["docs"]),
            "docs": x["docs"]
        })
        chain = fan | compose | {
            "answer": (qa_t | llm | StrOutputParser() | RunnableLambda(self.dedupe_lines)),
            "sources": RunnableLambda(lambda y: y["docs"]),
        }
        return chain


    def generate_single_mcq_from_context(self, context_text, llm, quiz_prompt):
        raw = (quiz_prompt | llm | StrOutputParser()).invoke({"context": context_text})
        return parse_quiz_json(raw)

    def build_quiz_items_from_topic(self, topic, search_tool, llm, quiz_prompt, n_questions=5):
        docs = search_tool.run(topic)
        shuffled = list(docs)
        random.shuffle(shuffled)
        items = []
        used_docs = []
        for d in shuffled:
            if len(items) >= n_questions:
                break

            context_text = (d.get("text") or "")
            parsed = self.generate_single_mcq_from_context(context_text, llm, quiz_prompt)

            if parsed:
                parsed["source"] = {
                    "book": d.get("book"),
                    "chunk_id": d.get("chunk_id"),
                    "granularity": d.get("granularity"),
                    "position": d.get("position"),
                }
                items.append(parsed)
                used_docs.append(d)
        return {"items": items, "sources": used_docs}

    def generate_quiz_payload(self, payload, search_tool, llm, quiz_prompt, n_questions):
        topic = payload["topic"]
        result = self.build_quiz_items_from_topic(topic, search_tool, llm, quiz_prompt, n_questions=n_questions)
        return {"items": result["items"], "sources": result["sources"]}
    
    def build_lc_quiz_chain(self, llm, search_tool, n_questions=5):
        quiz_prompt = self.lc_templates()[1]
        bound = partial(
            self.generate_quiz_payload,
            search_tool=search_tool,
            llm=llm,
            quiz_prompt=quiz_prompt,
            n_questions=n_questions,
        )
        return RunnableLambda(bound)
