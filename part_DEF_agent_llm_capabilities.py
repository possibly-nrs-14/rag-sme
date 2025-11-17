# part_DEF_agent_llm_capabilities.py
from functools import partial
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline, BitsAndBytesConfig
from langchain_huggingface import HuggingFacePipeline
from operator import itemgetter
import random
import logging
import os
import json
import random
from typing import Optional, Any, List, Dict, TypedDict, Annotated
from pydantic import PrivateAttr
from langchain_core.prompts import PromptTemplate
from langchain_core.tools import BaseTool, Tool
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda, RunnableParallel

# ---
from part_G_RAG import (
    EmbeddingIndex, ElasticsearchIndex, QueryEncoder, BGEReranker,
    SearchPipeline, load_elasticsearch_config
)
from custom_agent import AgentExecutor, ConversationMemory
from part_H_tools import EmailTool, ExportTool

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
    def __init__(self, model_name='google/medgemma-4b-it', max_new_tokens=256, temperature=0.05, prompt_strategy='zero-shot'):
        self.model_name = model_name
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.prompt_strategy = prompt_strategy
    def load_llm_lc(self):
        logger.info(f"Loading model: {self.model_name}")
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
            if self.prompt_strategy == 'zero-shot':
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
            else:
                qa = PromptTemplate.from_template(
                    "You are a neuroanatomy assistant.\n"
                    "Use ONLY the information in the Context to answer the Question.\n"
                    "You must output exactly ONE of the following:\n"
                    "1) A single, self-contained answer in at most 3 sentences of plain English, IF the Context clearly contains the answer.\n"
                    "2) Exactly the phrase: Insufficient context. (nothing else), IF the Context does NOT contain the answer.\n"
                    "Do NOT include backticks or markdown fences in your reply.\n\n"
                    "--- EXAMPLE 1 ---\n"
                    "Context: The cerebellum is located at the back of the brain, inferior to the cerebrum. It is responsible for coordinating voluntary movements.\n"
                    "Question: What is the function of the cerebellum?\n"
                    "Answer: The cerebellum is responsible for coordinating voluntary movements.\n\n"
                    "--- EXAMPLE 2 ---\n"
                    "Context: The cerebrum is the largest part of the brain. The frontal lobe is one of its four main lobes.\n"
                    "Question: What is the primary role of the hippocampus?\n"
                    "Answer: Insufficient context.\n"
                    "--- END EXAMPLES ---\n\n"
                    "Context:\n{context}\n\n"
                    "Question: {question}\n"
                    "Answer:"
                )
            quiz = PromptTemplate.from_template(
                "Create ONE MCQ (A–D) from the context. One correct option only. Return JSON with question, options (A..D), correct.\n\n"
                "--- EXAMPLE 1 ---\n"
                "Context: The central nervous system (CNS) consists of the brain and the spinal cord.\n"
                "JSON:\n"
                '{{\n'
                '  "question": "What are the two main components of the central nervous system (CNS)?",\n'
                '  "options": {{\n'
                '    "A": "The brain and the peripheral nerves",\n'
                '    "B": "The brain and the spinal cord",\n'
                '    "C": "The spinal cord and the cranial nerves",\n'
                '    "D": "The cerebrum and the cerebellum"\n'
                '  }},\n'
                '  "correct": "B"\n'
                '}}\n'
                "--- END EXAMPLES ---\n\n"
                "Context:\n{context}\n\nJSON:"
            )

        elif self.model_name == 'Intelligent-Internet/II-Medical-8B':
            if self.prompt_strategy == 'zero-shot':
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
            else:
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
                    "--- EXAMPLE 1 ---\n"
                    "Context: The cerebellum coordinates voluntary movements.\n"
                    "Question: What does the cerebellum do?\n"
                    "<|assistant|>\n"
                    "**Reasoning:**\n"
                    "[The user is asking for the function of the cerebellum. The context states 'The cerebellum coordinates voluntary movements'. This directly answers the question.]\n\n"
                    "**Final Answer:**\n"
                    "[The cerebellum coordinates voluntary movements.]\n\n"
                    "<|user|>\n"
                    "--- EXAMPLE 2 ---\n"
                    "Context: The frontal lobe is involved in planning.\n"
                    "Question: What is the function of the temporal lobe?\n"
                    "<|assistant|>\n"
                    "**Reasoning:**\n"
                    "[The user is asking about the temporal lobe. The context only mentions the frontal lobe. Therefore, the context is insufficient.]\n\n"
                    "**Final Answer:**\n"
                    "[Insufficient context.]\n\n"
                    "<|user|>\n"
                    "--- END EXAMPLES ---\n\n"
                    "**Context:**\n"
                    "{context}\n\n"
                    "**Question:**\n"
                    "{question}\n\n"
                    "<|assistant|>\n"
                )
            quiz = PromptTemplate.from_template(
                    "<|system|>\n"
                    "Create ONE MCQ (A–D) from the context. One correct option only. Return *only* a valid JSON object with question, options (A..D), correct.\n"
                    "<|user|>\n"
                    "--- EXAMPLE 1 ---\n"
                    "Context: The central nervous system (CNS) consists of the brain and the spinal cord.\n"
                    "<|assistant|>\n"
                    '{{\n'
                    '  "question": "What are the two main components of the central nervous system (CNS)?",\n'
                    '  "options": {{\n'
                    '    "A": "The brain and the peripheral nerves",\n'
                    '    "B": "The brain and the spinal cord",\n'
                    '    "C": "The spinal cord and the cranial nerves",\n'
                    '    "D": "The cerebrum and the cerebellum"\n'
                    '  }},\n'
                    '  "correct": "B"\n'
                    '}}\n'
                    "<|user|>\n"
                    "--- END EXAMPLES ---\n\n"
                    "Context:\n{context}\n\n"
                    "<|assistant|>"
                )
        elif self.model_name == 'microsoft/MediPhi':
            if self.prompt_strategy == 'zero-shot':
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
            else:
                qa = PromptTemplate.from_template(
                    "<|system|>\n"
                    "You are an expert clinical QA assistant.\n"
                    "You must answer the user's question based *only* on the provided context.\n"
                    "- If the context contains the answer, provide a concise, single-paragraph answer.\n"
                    "- If the context does NOT contain the answer, you MUST respond with *only* the exact phrase: Insufficient context.\n"
                    "<|end|>\n"
                    "<|user|>\n"
                    "--- EXAMPLE 1 ---\n"
                    "Context: The cerebellum coordinates voluntary movements.\n"
                    "Question: What does the cerebellum do?\n"
                    "<|end|>\n"
                    "<|assistant|>\n"
                    "The cerebellum coordinates voluntary movements.\n"
                    "<|end|>\n"
                    "<|user|>\n"
                    "--- EXAMPLE 2 ---\n"
                    "Context: The frontal lobe is involved in planning.\n"
                    "Question: What is the function of the temporal lobe?\n"
                    "<|end|>\n"
                    "<|assistant|>\n"
                    "Insufficient context.\n"
                    "<|end|>\n"
                    "<|user|>\n"
                    "--- END EXAMPLES ---\n\n"
                    "Context:\n"
                    "{context}\n\n"
                    "Question: {question}\n"
                    "<|end|>\n"
                    "<|assistant|>"
                )
            quiz = PromptTemplate.from_template(
                    "<|system|>\n"
                    "You are a quiz generation bot. Create ONE MCQ (A–D) from the context.\n"
                    "The correct option must be from the context.\n"
                    "You must return *only* a single valid JSON object with 'question', 'options' (A,B,C,D), and 'correct'.\n"
                    "<|end|>\n"
                    "<|user|>\n"
                    "Context: The central nervous system (CNS) consists of the brain and the spinal cord.\n"
                    "<|end|>\n"
                    "<|assistant|>\n"
                    '{{\n'
                    '  "question": "What are the two main components of the central nervous system (CNS)?",\n'
                    '  "options": {{\n'
                    '    "A": "The brain and the peripheral nerves",\n'
                    '    "B": "The brain and the spinal cord",\n'
                    '    "C": "The spinal cord and the cranial nerves",\n'
                    '    "D": "The cerebrum and the cerebellum"\n'
                    '  }},\n'
                    '  "correct": "B"\n'
                    '}}\n'
                    "<|end|>\n"
                    "<|user|>\n"
                    "Context:\n"
                    "{context}\n"
                    "<|end|>\n"
                    "<|assistant|>"
                )
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
    
    def build_agent_executor(self, llm, search_tool, max_iterations=5):

        # Build the QA and Quiz chains
        qa_chain = self.build_lc_qa_chain(llm, search_tool)
        quiz_chain = self.build_lc_quiz_chain(llm, search_tool, n_questions=5)
        
        # Create tools list
        tools = [
            search_tool,
            Tool(
                name="answer_question_about_neuroanatomy",
                func=qa_chain.invoke,
                description="""Use this tool to answer a specific question about neuroanatomy, the human nervous system, or related medical topics.
                Input must be a JSON object with a 'question' key, e.g.: {"question": "What is the function of the cerebellum?"}
                The output will be a JSON object containing the 'answer' and 'sources'."""
            ),
            Tool(
                name="generate_quiz_on_neuroanatomy_topic",
                func=quiz_chain.invoke,
                description="""Use this tool to generate a multiple-choice quiz on a given neuroanatomy topic.
            Input must be a JSON object with a 'topic' key, e.g.: {"topic": "The frontal lobe"}
            The output will be a JSON object containing a list of 'items'."""
                    ),
            ExportTool(),
            EmailTool()
        ]
        executor = AgentExecutor(
            llm=llm,
            model_name=self.model_name,
            tools=tools,
            max_iterations=max_iterations,
            verbose=True 
        )
        
        logger.info("Agent executor created successfully.")
        return executor

    def invoke_agent(self, agent_executor, user_input, memory=None):
        chat_history = ""
        if memory:
            chat_history = memory.get_history_string()
        
        # Invoke the executor
        result = agent_executor.invoke({
            "input": user_input,
            "chat_history": chat_history
        })
        
        return result