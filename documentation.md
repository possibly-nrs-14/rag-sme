# Nervous System SME - Complete System Documentation

## Table of Contents
1. [System Overview](#system-overview)
2. [Architecture & Workflow](#architecture--workflow)
3. [Component Breakdown](#component-breakdown)
4. [Prompting Strategies](#prompting-strategies)
5. [Frontend Integration](#frontend-integration)
6. [Workflow Diagram](#workflow-diagram)

---

## System Overview

The Nervous System Subject Matter Expert (SME) is a comprehensive RAG (Retrieval-Augmented Generation) system designed to provide intelligent question-answering and quiz generation capabilities for neuroanatomy content. The system processes medical textbooks, creates semantic embeddings, and uses large language models to answer questions and generate educational quizzes.

### Key Features
- **Multi-format Document Processing**: Supports PDF, DOCX, PPTX, TXT, and MD files
- **Multi-granularity Chunking**: Processes documents at 2048, 512, and 128 token granularities
- **Hybrid Search**: Combines vector similarity (semantic) and BM25 (keyword) search
- **Elasticsearch Integration**: Optional production-grade search backend with FAISS fallback
- **Agentic Chat Interface**: Conversational agent capable of multi-step reasoning
- **Document Export**: Generates PDF, DOCX, and PPTX exports for Q&A and quizzes
- **Dynamic LLM Configuration**: Switch models and prompting strategies without restart

---

## Architecture & Workflow

The system follows a modular pipeline architecture:

```
Data Ingestion → Preprocessing → Embedding → Indexing → Retrieval → LLM Generation → Export
```

### High-Level Flow

1. **Collection (Part A)**: Discovers and catalogs documents
2. **Preprocessing (Part B)**: Parses, cleans, chunks, and deduplicates content
3. **Embedding (Part C)**: Generates semantic embeddings using domain-specific models
4. **RAG System (Part G)**: Implements hybrid search with reranking
5. **LLM Capabilities (Part D/E/F)**: Provides QA and quiz generation chains
6. **Tools (Part H)**: Export and email functionality
7. **System Integration (Part I)**: FastAPI web application with chat and task interfaces

---

## Component Breakdown

### Part A: Collection (`part_A_collection.py`)

**Purpose**: Document discovery and metadata generation

**Key Functions**:
- `collect_and_organize_documents(input_dir)`: Recursively scans for supported file types (PDF, DOCX, PPTX, TXT, MD)
- Organizes files by extension for batch processing
- Generates metadata registry with file names, types, and ingestion timestamps

**Approach**:
- Uses `glob` with recursive pattern matching to find all documents
- Creates a structured metadata JSON file for tracking document provenance
- Provides summary statistics for validation

**Output**: `metadata/file_metadata.json`

---

### Part B: Preprocessing (`part_B_preprocessing.py`)

**Purpose**: Text extraction, cleaning, chunking, and deduplication

**Key Components**:

#### 1. **Multi-Format Parsing**
- **PDF**: Uses PyMuPDF (`fitz`) with column-aware block extraction
  - Detects left/right columns by calculating page midpoint
  - Orders blocks left-to-right, top-to-bottom for proper reading flow
  - Implements chapter gating to skip front matter and index sections
- **DOCX**: Extracts paragraphs using `python-docx`
- **PPTX**: Extracts text from slide shapes
- **TXT/MD**: Direct file reading with encoding handling

#### 2. **Content Cleaning**
- **Whitespace Normalization**: Standardizes spaces, tabs, and newlines via `helpers.normalize_spaces()`
- **Chapter Detection**: Uses pattern matching to identify chapter starts (e.g., "chapter", "gross anatomy of the brain")
- **Index Trimming**: Detects and removes index sections at document end
- **Minimum Word Threshold**: Filters out blocks with fewer than 10 words (removes headers, footers, page numbers)

#### 3. **Content-Aware Chunking**
The chunking strategy combines paragraph-aware splitting with recursive token-based splitting:

- **Paragraph Splitting**: Splits on double newlines, preserving paragraph boundaries
- **Token Counting**: Uses regex-based tokenization (`\w+|[^\w\s]`) for accurate token counts
- **Overlap Strategy**: Implements context-aware overlap:
  - For 2048 tokens: 256 token overlap
  - For 512 tokens: 128 token overlap
  - For 128 tokens: 32 token overlap
- **Recursive Fallback**: If a paragraph exceeds max_tokens, recursively splits it with overlap
- **Paragraph Aggregation**: Combines paragraphs until approaching token limit, then creates new chunk with overlap

#### 4. **Deduplication**
- **MinHash LSH**: Uses Locality-Sensitive Hashing for near-duplicate detection
- **Threshold**: 0.9 similarity threshold (90% similarity considered duplicate)
- **Two-Level Deduplication**:
  - Per-document: Removes duplicates within a single document
  - Cross-document: Batch-level deduplication across all documents at maximum granularity

#### 5. **Sanitization**
- **Prompt Injection Defense**: Implements defense-in-depth sanitization
  - Strips non-printable control characters
  - Replaces common injection patterns (e.g., "ignore previous instructions", "you are now", ChatML markers)
  - Applied both during block cleaning and before final chunk emission

**Output Structure**:
```
artifacts/
  clean/
    <book>__clean.txt
  chunks/
    <granularity>_tokens_<book>.jsonl
    <granularity>_tokens_batch_deduped.jsonl
  logs/
    ingestion_<timestamp>.log
```

---

### Part C: Embeddings (`part_C_embeddings.py`)

**Purpose**: Generate semantic embeddings and build document graphs

**Key Components**:

#### 1. **TextEmbedder Class**
- **Model**: Uses `FremyCompany/BioLORD-2023` (domain-specific biomedical embedding model)
  - Better performance than general-purpose models (e.g., `all-mpnet-base-v2`) on medical benchmarks
- **Normalization**: L2-normalizes embeddings for cosine similarity computation
- **Batch Processing**: Configurable batch size (default 128) for efficient GPU utilization
- **Fallback**: Deterministic hashing fallback if model unavailable

#### 2. **Document Graph Construction**
- **Node Types**:
  - **Document Node**: Represents the parent document (no embedding)
  - **Chunk Nodes**: Individual text chunks with embeddings and metadata
- **Edge Types**:
  - **child_of**: Links chunks to parent document
  - **next**: Sequential relationships between chunks
- **Elasticsearch Integration**: Optionally indexes chunks to Elasticsearch during graph creation

**Output Structure**:
```
artifacts/graph/
  <granularity>_tokens_<stem>__nodes.jsonl
  <granularity>_tokens_<stem>__edges.jsonl
```

**Design Decisions**:
- Graph structure enables hierarchical queries and context traversal
- Embeddings stored with nodes for efficient retrieval
- Supports multiple granularities for different query types (detailed vs. overview)

---

### Part G: RAG System (`part_G_RAG.py`)

**Purpose**: Retrieval system with hybrid search and reranking

**Key Components**:

#### 1. **Dual Backend Support**

**EmbeddingIndex (FAISS)**:
- In-memory vector index using FAISS `IndexFlatIP` (Inner Product = cosine similarity on normalized vectors)
- Loads embeddings from graph nodes or embedding JSONL files
- Fast, suitable for development and small-to-medium corpora

**ElasticsearchIndex**:
- Production-grade search backend
- Supports vector search, BM25 keyword search, and hybrid search
- Configurable via `config/elasticsearch.json`
- Automatic fallback to FAISS if Elasticsearch unavailable

#### 2. **Search Modes**

**Vector Search**:
- Semantic similarity using cosine distance on embeddings
- Encodes query using same embedding model (BioLORD-2023)
- Returns top-k most semantically similar chunks

**BM25 Search**:
- Keyword-based search using Elasticsearch's BM25 algorithm
- Better for exact term matching and technical terminology
- Uses English analyzer for stemming and tokenization

**Hybrid Search**:
- Combines vector and BM25 results using Reciprocal Rank Fusion (RRF)
- Formula: `RRF_score = α/(k + rank_vec) + (1-α)/(k + rank_bm25)`
- Default α = 0.5 (balanced), k = 60
- Provides best of both worlds: semantic understanding + keyword precision

#### 3. **Reranking**

**BGEReranker**:
- Primary: `BAAI/bge-reranker-base` via FlagEmbedding library
- Fallback: `cross-encoder/ms-marco-MiniLM-L-6-v2` via sentence-transformers
- Cross-encoder architecture for query-document relevance scoring
- Auto-detects GPU/CPU and uses FP16 on GPU, FP32 on CPU

**SearchPipeline**:
- Unified interface for both backends
- Retrieves candidates (default 120), reranks to top-k (default 8)
- Supports metadata filtering (e.g., filter by source book, granularity)

**Design Decisions**:
- Hybrid search addresses limitations of pure vector search (misspellings, exact terms)
- Reranking improves precision by re-scoring top candidates
- Configurable search mode allows optimization for different query types

---

### Part D/E/F: LLM Capabilities (`part_DEF_agent_llm_capabilities.py`)

**Purpose**: LLM integration for QA and quiz generation

**Key Components**:

#### 1. **LLMAgent Class**

**Supported Models**:
- `google/medgemma-4b-it`: Google's medical instruction-tuned model
- `microsoft/MediPhi`: Microsoft's clinical QA model
- `Intelligent-Internet/II-Medical-8B`: Medical domain model with 8B parameters

**Configuration**:
- **Temperature**: Default 0.05 (low for factual accuracy)
- **Max Tokens**: Default 512 (increased from 256 to prevent truncation)
- **Quantization**: 4-bit quantization via BitsAndBytesConfig for memory efficiency
- **Device**: Auto device mapping (GPU if available)

#### 2. **SearchDocsTool**

LangChain tool wrapper for semantic search:
- Accepts string queries or JSON with filters
- Returns top-k results with scores and metadata
- Integrates with agent executor for tool calling

#### 3. **QA Chain**

**Architecture**:
```
Query → SearchDocsTool → Context Aggregation → LLM → Answer
```

**Process**:
1. User question triggers search over corpus
2. Top 3 chunks aggregated with separators
3. Context + question passed to LLM with prompt template
4. LLM generates answer constrained to context
5. Returns answer + source citations

**Output Format**:
- Answer: Concise 1-3 sentence response
- Sources: List of chunks with book, chunk_id, granularity, position
- Special handling: Returns "Insufficient context." if answer not in context

#### 4. **Quiz Generation Chain**

**Process**:
1. Topic query triggers search
2. Shuffles results for diversity
3. For each chunk, generates MCQ using LLM
4. Parses JSON response: `{question, options: {A, B, C, D}, correct}`
5. Continues until n_questions generated or chunks exhausted

**Validation**:
- Ensures exactly 4 options (A-D)
- Validates correct answer is one of the options
- Skips malformed questions

---

### Part H: Tools (`part_H_tools.py`)

**Purpose**: Document export and email functionality

**Key Components**:

#### 1. **ExportTool**

**Supported Formats**:
- **PDF**: Using ReportLab for professional formatting
- **DOCX**: Using python-docx for Word documents
- **PPTX**: Using python-pptx for PowerPoint presentations (quiz only)

**Export Types**:
- **QA Export**: Question, answer, and source citations
- **Quiz Export**: Questions with options, answer key, and sources

**Features**:
- Automatic page breaks for long content
- Text wrapping for readability
- Timestamped filenames to prevent overwrites
- Sources section with chunk metadata

#### 2. **EmailTool**

**Purpose**: Placeholder for email functionality (logs instead of sending per project spec)

**Design**: Structured payload with recipient, subject, body, and optional attachment

---

### Part I: System Components (`part_I_system_components.py`)

**Purpose**: FastAPI web application integrating all components

**Key Features**:

#### 1. **Startup Initialization**
- Conditional ingestion based on `SME_INGEST_ON_START` environment variable
- Automatic Elasticsearch index creation if enabled
- Search tool initialization with configurable mode (vector/bm25/hybrid)
- LLM component lazy loading via config manager

#### 2. **API Endpoints**

**Frontend Routes**:
- `GET /`: Redirects to `/chat`
- `GET /chat`: Chat interface template
- `GET /tools`: QA/Quiz tools interface template

**Direct LangChain Endpoints**:
- `POST /lc/qa`: Direct QA chain invocation (bypasses agent)
- `POST /lc/quiz`: Direct quiz generation (bypasses agent)

**Chat Endpoint**:
- `POST /chat`: Conversational agent with memory
  - Maintains session-based conversation history
  - Returns intermediate steps for transparency
  - Handles complex multi-step queries

**Export Endpoints**:
- `POST /export/qa`: Export QA to PDF/DOCX
- `POST /export/quiz`: Export quiz to PDF/DOCX/PPTX

**Admin Endpoints**:
- `POST /admin/ingest`: Trigger manual ingestion
- `GET /admin/search-backend`: Check search backend status
- `POST /admin/es/create-index`: Create Elasticsearch index
- `POST /admin/es/reindex`: Reindex all chunks
- `POST /admin/llm-config`: Update LLM configuration dynamically
- `GET /admin/llm-config`: Get current LLM configuration

#### 3. **Session Management**

**ConversationMemory**:
- Per-session conversation history (default max 10 turns)
- Stores user messages and assistant responses
- Provides history string for agent context
- Supports clearing history via `/chat/clear`

---

### Helper Modules

#### `helpers.py`

**Functions**:

1. **`normalize_spaces(s)`**:
   - Collapses multiple spaces/tabs to single space
   - Normalizes non-breaking spaces (U+00A0)
   - Removes leading/trailing whitespace around newlines
   - Limits consecutive newlines to maximum of 2

2. **`sanitize_for_injection(text)`**:
   - **Defense-in-Depth**: Applied at multiple stages (block cleaning, chunk emission, query sanitization)
   - Removes non-printable control characters
   - Replaces injection patterns:
     - Instruction overrides: "ignore previous instructions", "forget your instructions"
     - Role-playing: "you are now", "your new instructions are"
     - ChatML markers: `<|system|>`, `<|user|>`, `<|assistant|>`
     - Evasion attempts: "stop being a chatbot", "reveal your instructions"
   - Uses regex compilation for efficiency
   - Returns sanitized text with `[SANITIZED_INSTRUCTION]` markers

**Design Philosophy**: Multiple sanitization points prevent injection at different stages (source documents, user queries, chunk boundaries)

---

#### `llm_config.py`

**Purpose**: Dynamic LLM configuration management without server restart

**Key Components**:

1. **LLMConfig Dataclass**:
   - Model name, temperature, max_new_tokens, prompt_strategy
   - Validation ensures allowed models and strategies

2. **LLMConfigManager**:
   - Thread-safe singleton for configuration
   - Caches LLM agent, chains, and executor
   - **Memory Management**: Explicitly unloads old models when switching
     - Deletes cached objects
     - Forces garbage collection (3 passes)
     - Clears CUDA cache if available
   - Lazy loading: Creates components on first request

**Allowed Models**:
- `google/medgemma-4b-it`
- `microsoft/MediPhi`
- `Intelligent-Internet/II-Medical-8B`

**Allowed Strategies**:
- `zero-shot`: Direct instructions
- `few-shot`: Includes examples

**API Integration**: FastAPI endpoints allow runtime configuration changes

---

#### `custom_agent.py`

**Purpose**: Custom agent executor implementing ReAct (Reasoning + Acting) pattern

**Key Components**:

1. **AgentExecutor Class**:

**Architecture**:
- Implements iterative Thought-Action-Observation loop
- Maximum iterations: 5 (configurable)
- Model-specific prompt templates for different LLMs

**Prompt Structure**:
```
System: Role definition + tool descriptions + rules
User: Chat history + current input + agent scratchpad
Assistant: Thought → Action → Action Input → Observation (repeat)
```

**Tool Execution**:
- Parses LLM output for Action/Action Input
- Executes tool and captures observation
- Updates scratchpad for next iteration
- Returns final answer when LLM indicates completion

**Special Handling**:
- **Export Tool Normalization**: Automatically repairs incomplete export payloads
  - Extracts data from previous QA/quiz observations
  - Completes missing fields (question, topic, sources)
  - Allows agent to use simplified export commands

**Model-Specific Prompts**:
- **MediPhi/II-Medical-8B**: ChatML format with `<|system|>`, `<|user|>`, `<|assistant|>` markers
- **MedGemma**: Plain text format

2. **ConversationMemory Class**:
- Maintains conversation history per session
- Trims to max_history * 2 (user + assistant pairs)
- Provides formatted history string for agent context

**Design Decisions**:
- Custom executor provides more control than LangChain's default
- Explicit intermediate steps enable debugging and transparency
- Export normalization reduces agent errors and improves UX

---

## Prompting Strategies

The system implements two primary prompting strategies: **zero-shot** and **few-shot**. These strategies are model-specific, as different LLMs require different prompt formats and structures.

### Zero-Shot Prompting

**Definition**: Direct instructions without examples, relying on the model's pre-training to understand the task.

**Use Cases**: 
- Faster inference (shorter prompts)
- When model has strong instruction-following capabilities
- For simpler, well-defined tasks

**Implementation by Model**:

#### 1. **MedGemma-4B-IT** (Zero-Shot)

**QA Template**:
```
You are a neuroanatomy assistant.
Use ONLY the information in the Context to answer the Question.
You must output exactly ONE of the following:
1) A single, self-contained answer in at most 3 sentences of plain English, 
   IF the Context clearly contains the answer.
2) Exactly the phrase: Insufficient context. (nothing else), 
   IF the Context does NOT contain the answer.
You MUST NOT output both an answer AND 'Insufficient context.'
Do NOT write code, pseudo-code, or functions.
Do NOT include backticks or markdown fences in your reply.

Context:
{context}

Question: {question}
Answer:
```

**Key Features**:
- Explicit binary decision structure
- Clear output format constraints
- Prohibits markdown/code formatting
- Emphasizes context-only answering

**Quiz Template**:
```
Create ONE MCQ (A–D) from the context. One correct option only. 
Return JSON with question, options (A..D), correct.

Context:
{context}

JSON:
```

#### 2. **II-Medical-8B** (Zero-Shot)

**QA Template** (ChatML format):
```
<|system|>
You are an expert neuroanatomy assistant. Your task is to answer the user's question based *only* on the provided context.
Follow these steps:
1. Carefully read the Question and the Context.
2. Reason step-by-step to determine if the Context contains the information to answer the Question.
3. If the answer is in the context, formulate a concise, one-paragraph answer.
4. If the answer is NOT in the context, your final answer MUST be exactly: Insufficient context.
5. Provide your reasoning and final answer in the specified format.

<|user|>
**Context:**
{context}

**Question:**
{question}

<|assistant|>
```

**Key Features**:
- Step-by-step reasoning instructions
- ChatML format for structured dialogue
- Encourages explicit reasoning process
- Maintains context-only constraint

#### 3. **MediPhi** (Zero-Shot)

**QA Template** (ChatML format):
```
<|system|>
You are an expert clinical QA assistant.
You must answer the user's question based *only* on the provided context.
- If the context contains the answer, provide a concise, single-paragraph answer.
- If the context does NOT contain the answer, you MUST respond with *only* the exact phrase: Insufficient context.
<|end|>
<|user|>
Context:
{context}

Question: {question}
<|end|>
<|assistant|>
```

**Key Features**:
- Clinical focus in system message
- Explicit `<|end|>` markers for turn boundaries
- Concise instruction style
- Strong emphasis on context-only answering

---

### Few-Shot Prompting

**Definition**: Includes examples in the prompt to demonstrate desired behavior and output format.

**Use Cases**:
- When model needs guidance on format
- For complex tasks requiring specific structure
- To improve consistency and reduce errors

**Advantages**:
- **Format Learning**: Models learn exact output structure from examples
- **Error Reduction**: Examples demonstrate edge cases (e.g., "Insufficient context")
- **Consistency**: Reduces variability in responses

**Disadvantages**:
- **Longer Prompts**: Increases token usage and inference time
- **Context Window**: Takes up space that could be used for more context
- **Overfitting Risk**: Models may overfit to example style

**Implementation by Model**:

#### 1. **MedGemma-4B-IT** (Few-Shot)

**QA Template**:
```
You are a neuroanatomy assistant.
Use ONLY the information in the Context to answer the Question.
You must output exactly ONE of the following:
1) A single, self-contained answer in at most 3 sentences of plain English, IF the Context clearly contains the answer.
2) Exactly the phrase: Insufficient context. (nothing else), IF the Context does NOT contain the answer.
Do NOT include backticks or markdown fences in your reply.

--- EXAMPLE 1 ---
Context: The cerebellum is located at the back of the brain, inferior to the cerebrum. It is responsible for coordinating voluntary movements.
Question: What is the function of the cerebellum?
Answer: The cerebellum is responsible for coordinating voluntary movements.

--- EXAMPLE 2 ---
Context: The cerebrum is the largest part of the brain. The frontal lobe is one of its four main lobes.
Question: What is the primary role of the hippocampus?
Answer: Insufficient context.
--- END EXAMPLES ---

Context:
{context}

Question: {question}
Answer:
```

**Key Features**:
- Two examples: one with answer, one with "Insufficient context"
- Demonstrates proper format and decision logic
- Clear example boundaries with markers

**Quiz Template** (Few-Shot):
```
Create ONE MCQ (A–D) from the context. One correct option only. Return JSON with question, options (A..D), correct.

--- EXAMPLE 1 ---
Context: The central nervous system (CNS) consists of the brain and the spinal cord.
JSON:
{
  "question": "What are the two main components of the central nervous system (CNS)?",
  "options": {
    "A": "The brain and the peripheral nerves",
    "B": "The brain and the spinal cord",
    "C": "The spinal cord and the cranial nerves",
    "D": "The cerebrum and the cerebellum"
  },
  "correct": "B"
}
--- END EXAMPLES ---

Context:
{context}

JSON:
```

#### 2. **II-Medical-8B** (Few-Shot)

**QA Template**:
```
<|system|>
You are an expert neuroanatomy assistant. Your task is to answer the user's question based *only* on the provided context.
[... instructions ...]

<|user|>
--- EXAMPLE 1 ---
Context: The cerebellum coordinates voluntary movements.
Question: What does the cerebellum do?
<|assistant|>
**Reasoning:**
[The user is asking for the function of the cerebellum. The context states 'The cerebellum coordinates voluntary movements'. This directly answers the question.]

**Final Answer:**
[The cerebellum coordinates voluntary movements.]

<|user|>
--- EXAMPLE 2 ---
Context: The frontal lobe is involved in planning.
Question: What is the function of the temporal lobe?
<|assistant|>
**Reasoning:**
[The user is asking about the temporal lobe. The context only mentions the frontal lobe. Therefore, the context is insufficient.]

**Final Answer:**
[Insufficient context.]

<|user|>
--- END EXAMPLES ---

**Context:**
{context}

**Question:**
{question}

<|assistant|>
```

**Key Features**:
- Includes reasoning step in examples
- Shows structured output format
- Demonstrates both answer and insufficient context cases
- Maintains ChatML structure

#### 3. **MediPhi** (Few-Shot)

**QA Template**:
```
<|system|>
You are an expert clinical QA assistant.
[... instructions ...]
<|end|>
<|user|>
--- EXAMPLE 1 ---
Context: The cerebellum coordinates voluntary movements.
Question: What does the cerebellum do?
<|end|>
<|assistant|>
The cerebellum coordinates voluntary movements.
<|end|>
<|user|>
--- EXAMPLE 2 ---
Context: The frontal lobe is involved in planning.
Question: What is the function of the temporal lobe?
<|end|>
<|assistant|>
Insufficient context.
<|end|>
<|user|>
--- END EXAMPLES ---

Context:
{context}

Question: {question}
<|end|>
<|assistant|>
```

**Key Features**:
- Minimal examples showing direct answers
- Emphasizes exact phrase for "Insufficient context"
- Uses `<|end|>` markers consistently
- Concise example format

---

### Prompting Strategy Comparison

#### Performance Analysis

| Aspect | Zero-Shot | Few-Shot |
|--------|-----------|----------|
| **Token Usage** | Lower (~200-400 tokens) | Higher (~600-1000 tokens) |
| **Inference Speed** | Faster | Slower (due to longer prompts) |
| **Format Consistency** | Variable (depends on model) | Higher (examples guide format) |
| **Error Rate** | Moderate (model may misinterpret) | Lower (examples reduce ambiguity) |
| **Context Window** | More space for context | Less space (examples consume tokens) |
| **Setup Complexity** | Simple | Requires crafting good examples |

#### Use Case Recommendations

**Zero-Shot Best For**:
- Models with strong instruction-following (e.g., MedGemma-4B-IT)
- When context is long and every token counts
- Fast inference requirements
- Simple, well-defined tasks

**Few-Shot Best For**:
- Models needing format guidance (e.g., smaller models)
- Complex output structures (e.g., JSON quiz generation)
- Reducing "Insufficient context" false positives
- Improving consistency across queries

#### Model-Specific Observations

**MedGemma-4B-IT**:
- Strong zero-shot performance due to instruction tuning
- Few-shot helps with JSON parsing for quizzes
- Examples reduce malformed JSON responses

**II-Medical-8B**:
- Benefits from few-shot reasoning examples
- Step-by-step structure in examples improves answer quality
- ChatML format requires careful example formatting

**MediPhi**:
- Zero-shot sufficient for simple QA
- Few-shot improves edge case handling
- Minimal examples effective (doesn't need verbose reasoning)

#### Implementation Strategy

The system allows **dynamic switching** between strategies via the LLM configuration API:

```python
# Zero-shot
POST /admin/llm-config
{
  "prompt_strategy": "zero-shot"
}

# Few-shot
POST /admin/llm-config
{
  "prompt_strategy": "few-shot"
}
```

**Recommendation**: Start with zero-shot for speed, switch to few-shot if:
- JSON parsing errors occur (quiz generation)
- Inconsistent "Insufficient context" responses
- Format violations in answers

---

## Frontend Integration

### FastAPI Architecture

The system uses **FastAPI** as the web framework, providing:
- Automatic API documentation (Swagger/OpenAPI)
- Pydantic request validation
- Async support for concurrent requests
- Jinja2 templating for HTML pages

### Static Files (`static/app.js`)

**Key Functionality**:

1. **LLM Configuration Management**:
   - Loads/saves config to localStorage
   - Syncs with server on page load
   - Applies configuration via `/admin/llm-config` endpoint
   - Shows model loading status (pending vs. active)

2. **Chat Interface** (`/chat`):
   - Real-time message rendering
   - Handles JSON responses (quiz/QA objects)
   - Displays intermediate steps in collapsible details
   - Session management with unique session IDs
   - Clear history functionality

3. **Tools Interface** (`/tools`):
   - Direct QA chain invocation
   - Quiz generation with configurable question count
   - Export buttons for PDF/DOCX/PPTX
   - Source citation display

4. **Toast Notifications**:
   - Non-blocking success/error messages
   - Replaces `window.alert()` for better UX

### Templates

#### `templates/chat.html`

**Structure**:
- LLM configuration bar at top
- Chat message container with scroll
- Input form with Enter/Shift+Enter handling
- Clear history button

**Features**:
- Welcome message on load
- Thinking indicator during requests
- Markdown-like rendering (bold, italic, code)
- JSON object rendering (quiz items, QA with sources)

#### `templates/tools.html`

**Structure**:
- Shared LLM configuration bar
- Two-column layout:
  - Left: QA Tool panel
  - Right: Quiz Tool panel

**Features**:
- Direct LangChain chain calls (bypass agent)
- Export functionality for both QA and quizzes
- Source citation lists
- Question count selector for quizzes

### API Communication

**Request Flow**:
```javascript
// Chat request
POST /chat
{
  "query": "Generate a quiz on the frontal lobe and export it as PDF",
  "session_id": "session_1234567890"
}

// Response
{
  "response": { /* quiz object or answer */ },
  "session_id": "session_1234567890",
  "tools_used": 2,
  "intermediate_steps": [
    {
      "thought": "I need to generate a quiz...",
      "action": "generate_quiz_on_neuroanatomy_topic",
      "action_input": {"topic": "frontal lobe"},
      "observation": { /* quiz items */ }
    },
    {
      "thought": "Now I should export it...",
      "action": "export_document",
      "action_input": {"export_type": "quiz", "file_format": "pdf"},
      "observation": "./exports/quiz_export_1234567890.pdf"
    }
  ]
}
```

**Error Handling**:
- Try-catch blocks around all API calls
- User-friendly error messages
- Toast notifications for failures
- Graceful degradation (e.g., fallback to FAISS if ES unavailable)

---

## Workflow Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│                    SYSTEM STARTUP                               │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
        ┌─────────────────────────────────────┐
        │  Part A: Collection                │
        │  - Scan ./data for documents       │
        │  - Generate metadata               │
        └─────────────────────────────────────┘
                              │
                              ▼
        ┌─────────────────────────────────────┐
        │  Part B: Preprocessing              │
        │  - Parse PDF/DOCX/PPTX/TXT          │
        │  - Clean & normalize                │
        │  - Chunk at 2048/512/128 tokens     │
        │  - Deduplicate (LSH)                │
        │  - Sanitize for injection           │
        └─────────────────────────────────────┘
                              │
                              ▼
        ┌─────────────────────────────────────┐
        │  Part C: Embeddings                 │
        │  - Generate embeddings (BioLORD)     │
        │  - Build document graph             │
        │  - Index to Elasticsearch (opt)     │
        └─────────────────────────────────────┘
                              │
                              ▼
        ┌─────────────────────────────────────┐
        │  FastAPI Application Starts          │
        │  - Initialize search tool           │
        │  - Load LLM config                  │
        │  - Create agent executor            │
        └─────────────────────────────────────┘
                              │
                    ┌─────────┴─────────┐
                    │                   │
                    ▼                   ▼
        ┌──────────────────┐  ┌──────────────────┐
        │  Chat Interface  │  │  Tools Interface │
        │  (/chat)         │  │  (/tools)        │
        └──────────────────┘  └──────────────────┘
                    │                   │
                    ▼                   ▼
        ┌─────────────────────────────────────┐
        │  User Query                         │
        └─────────────────────────────────────┘
                    │
        ┌───────────┴───────────┐
        │                       │
        ▼                       ▼
┌───────────────┐      ┌───────────────┐
│  Chat Agent   │      │  Direct Chain │
│  (Multi-step) │      │  (Single-step)│
└───────────────┘      └───────────────┘
        │                       │
        └───────────┬───────────┘
                    │
                    ▼
        ┌─────────────────────────────────────┐
        │  Part G: RAG Retrieval             │
        │  - Encode query (BioLORD)           │
        │  - Hybrid search (vector + BM25)    │
        │  - Rerank (BGE)                     │
        │  - Return top-k chunks              │
        └─────────────────────────────────────┘
                    │
                    ▼
        ┌─────────────────────────────────────┐
        │  LLM Generation                      │
        │  - Build prompt (zero/few-shot)      │
        │  - Generate answer/quiz              │
        │  - Parse & validate output           │
        └─────────────────────────────────────┘
                    │
        ┌───────────┴───────────┐
        │                       │
        ▼                       ▼
┌───────────────┐      ┌───────────────┐
│  Return to   │      │  Export Tool  │
│  User         │      │  (PDF/DOCX/   │
│               │      │   PPTX)       │
└───────────────┘      └───────────────┘
```

### Detailed Query Flow (Chat Agent)

```
User: "Generate a quiz on cerebellum and export as PDF"
  │
  ▼
Agent Executor (Iteration 1)
  │
  ├─ Thought: "User wants a quiz on cerebellum, then export"
  ├─ Action: generate_quiz_on_neuroanatomy_topic
  ├─ Action Input: {"topic": "cerebellum"}
  │
  ▼
SearchDocsTool
  │
  ├─ Encode query: "cerebellum"
  ├─ Hybrid search (vector + BM25)
  ├─ Rerank top 120 → top 8
  └─ Return chunks
  │
  ▼
Quiz Chain
  │
  ├─ For each chunk:
  │   ├─ Build prompt (zero/few-shot)
  │   ├─ Generate MCQ JSON
  │   └─ Parse & validate
  └─ Return 5 questions
  │
  ▼
Agent Executor (Iteration 2)
  │
  ├─ Thought: "Quiz generated, now export as PDF"
  ├─ Action: export_document
  ├─ Action Input: {"export_type": "quiz", "file_format": "pdf"}
  │
  ▼
ExportTool
  │
  ├─ Extract quiz data from previous observation
  ├─ Generate PDF with ReportLab
  └─ Return file path
  │
  ▼
Agent Executor (Final)
  │
  ├─ Thought: "Task complete"
  ├─ Final Answer: "I've generated a quiz on the cerebellum and exported it as PDF: ./exports/quiz_export_1234567890.pdf"
  │
  ▼
Response to User
  │
  ├─ Quiz preview in chat
  ├─ Intermediate steps (collapsible)
  └─ Export confirmation
```

---

## Key Design Decisions & Rationale

### 1. **Multi-Granularity Chunking**

**Decision**: Process documents at 2048, 512, and 128 token granularities.

**Rationale**:
- **2048 tokens**: Detailed context for complex questions requiring extensive information
- **512 tokens**: Balanced granularity for most queries
- **128 tokens**: Fine-grained for specific fact retrieval

**Benefits**: Query routing can select appropriate granularity based on question complexity.

### 2. **Hybrid Search**

**Decision**: Combine vector (semantic) and BM25 (keyword) search using RRF.

**Rationale**:
- **Vector search**: Captures semantic similarity but may miss exact terms
- **BM25 search**: Excellent for technical terminology and exact matches
- **Hybrid**: Best of both worlds, especially for medical domain with precise terminology

**Evidence**: Medical queries often contain specific anatomical terms that benefit from keyword matching.

### 3. **Domain-Specific Embeddings**

**Decision**: Use `FremyCompany/BioLORD-2023` instead of general-purpose models.

**Rationale**:
- Trained on biomedical literature
- Better performance on medical benchmarks
- Improved understanding of medical terminology and relationships

### 4. **Defense-in-Depth Sanitization**

**Decision**: Apply sanitization at multiple stages (block cleaning, chunk emission, query processing).

**Rationale**:
- Prevents prompt injection from source documents
- Protects against malicious user queries
- Multiple layers reduce risk of bypass

### 5. **Custom Agent Executor**

**Decision**: Implement custom ReAct-style executor instead of using LangChain's default.

**Rationale**:
- **Control**: Fine-grained control over tool execution and error handling
- **Transparency**: Explicit intermediate steps for debugging
- **Flexibility**: Model-specific prompt templates
- **Export Normalization**: Automatic repair of incomplete export payloads

### 6. **Dynamic LLM Configuration**

**Decision**: Allow runtime model and strategy switching without server restart.

**Rationale**:
- **Experimentation**: Easy A/B testing of models and strategies
- **Resource Management**: Unload models when switching to free memory
- **Flexibility**: Adapt to different query types (zero-shot for speed, few-shot for accuracy)

### 7. **Session-Based Memory**

**Decision**: Maintain conversation history per session ID.

**Rationale**:
- Enables multi-turn conversations
- Context preservation across queries
- Supports complex, multi-step requests

### 8. **Elasticsearch with FAISS Fallback**

**Decision**: Support both Elasticsearch (production) and FAISS (development).

**Rationale**:
- **Elasticsearch**: Scalable, supports hybrid search, production-ready
- **FAISS**: Fast, no external dependencies, good for development
- **Fallback**: System works even if Elasticsearch unavailable

---

## Conclusion

The Nervous System SME represents a comprehensive RAG system with careful attention to:
- **Robustness**: Multi-format support, error handling, fallbacks
- **Performance**: Hybrid search, efficient embeddings, quantization
- **Security**: Prompt injection defense, input sanitization
- **Usability**: Dynamic configuration, transparent intermediate steps, export functionality
- **Flexibility**: Multiple models, prompting strategies, search backends

The system successfully combines traditional information retrieval (BM25) with modern semantic search (embeddings) and large language models to provide accurate, context-aware responses for neuroanatomy education.

---

## Appendix: Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `SME_INGEST_ON_START` | `1` | Run ingestion on startup (0 to skip) |
| `SME_GRANULARITIES` | `2048,512,128` | Comma-separated token granularities |
| `SME_OVERLAP_TOKENS` | `64` | Default overlap tokens |
| `SME_SKIP_METADATA` | `false` | Skip metadata generation |
| `USE_ELASTICSEARCH` | `true` | Use Elasticsearch backend |
| `SEARCH_MODE` | `hybrid` | Search mode: vector, bm25, hybrid |
| `PORT` | `8000` | FastAPI server port |

---

*Documentation generated for Nervous System SME Project*

