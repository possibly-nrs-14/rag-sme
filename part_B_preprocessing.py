import os
import re
import sys
import glob
import uuid
import time
import hashlib
import logging
import orjson
from tqdm import tqdm
import fitz 
from datasketch import MinHash, MinHashLSH  

try:
    from pptx import Presentation
except ImportError:
    Presentation = None

try:
    import docx
except ImportError:
    docx = None

from part_C_embeddings import save_document_graph, TextEmbedder, DEFAULT_EMBED_MODEL
from part_A_collection import collect_and_organize_documents

# Import sanitization and normalization functions
from helpers import normalize_spaces, sanitize_for_injection

# Regex for tokenization, needed by LSH function
_WORD_RE = re.compile(r"\w+|[^\w\s]", re.UNICODE)

# Module-level constants for document cleaning heuristics
CHAPTER_START_PATTERNS = ["c h a p t e r", "gross anatomy of the brain", "chapter", "april2013", "brain–machine interfaces"]
CHAPTER_END_PATTERNS = ["index", "i n d e x"]
MIN_WORDS_THRESHOLD = 10

def setup_logging(log_dir):
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, f"ingestion_{int(time.time())}.log")
    
    # Clear existing handlers
    root_logger = logging.getLogger()
    if root_logger.hasHandlers():
        root_logger.handlers.clear()
        
    logging.basicConfig(
        filename=log_path,
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )
    console = logging.StreamHandler(sys.stdout)
    console.setLevel(logging.INFO)
    console.setFormatter(logging.Formatter("%(levelname)s | %(message)s"))
    logging.getLogger().addHandler(console)
    return log_path

def tokenize(s):
    return _WORD_RE.findall(s)

def count_tokens(s):
    return len(tokenize(s))

# normalize_spaces is now imported from helpers.py

def load_plain_text_file(path):
    """Load and sanitize a plain-text or markdown file."""
    try:
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            raw = f.read()
    except Exception as e:
        logging.exception(f"Failed to read text file {path}: {e}")
        return ""
    text = normalize_spaces(raw)
    return sanitize_for_injection(text)

def load_docx(path):
    """Extract text from DOCX, one paragraph per line."""
    if docx is None:
        logging.warning("python-docx not installed; skipping DOCX file: %s", path)
        return ""
    try:
        d = docx.Document(path)
        paras = [normalize_spaces(p.text) for p in d.paragraphs if p.text.strip()]
        text = "\n\n".join(paras)
        return sanitize_for_injection(text)
    except Exception as e:
        logging.exception(f"Failed to parse DOCX {path}: {e}")
        return ""

def load_pptx(path):
    """Extract text from PPTX slides."""
    if Presentation is None:
        logging.warning("python-pptx not installed; skipping PPTX file: %s", path)
        return ""
    try:
        pres = Presentation(path)
        texts = []
        for slide in pres.slides:
            for shape in slide.shapes:
                if hasattr(shape, "text") and shape.text:
                    texts.append(normalize_spaces(shape.text))
        text = "\n\n".join(t for t in texts if t)
        return sanitize_for_injection(text)
    except Exception as e:
        logging.exception(f"Failed to parse PPTX {path}: {e}")
        return ""

def clean_and_parse_any(path):
    """Unified loader that routes to the appropriate parser based on file extension."""
    ext = os.path.splitext(path)[1].lower()
    try:
        if ext == '.pdf':
            return clean_and_parse_pdf(path)
        elif ext == '.docx':
            return load_docx(path)
        elif ext == '.pptx':
            return load_pptx(path)
        elif ext in ['.txt', '.md']:
            return load_plain_text_file(path)
        else:
            logging.warning(f"Unsupported file extension: {ext} for file {path}")
            return ""
    except Exception as e:
        logging.exception(f"Failed to parse {path}: {e}")
        return ""

def split_into_paragraphs(text, threshold=10):
    """Part B: Content-aware (paragraph-based) splitting."""
    paras = re.split(r"\n\s*\n", text)
    return [normalize_spaces(p) for p in paras if p and count_tokens(p) >= threshold]

def recursive_split_by_tokens(text, max_tokens, overlap_tokens):
    """Part B: Recursive character splitting with overlap."""
    toks = tokenize(text)
    if len(toks) <= max_tokens:
        return [text]
    
    chunks = []
    start = 0
    step = max_tokens - overlap_tokens
    while start < len(toks):
        end = min(len(toks), start + max_tokens)
        window = " ".join(toks[start:end])
        chunks.append(normalize_spaces(window))
        if end == len(toks):
            break
        start += step
    return chunks

def content_aware_chunk(text, max_tokens, overlap_tokens):
    """Main chunking strategy combining paragraph-splitting and recursive methods."""
    paras = split_into_paragraphs(text)
    chunks, current, cur_tokens = [], [], 0

    for p in paras:
        ptoks = count_tokens(p)
        if ptoks > max_tokens:
            if current:
                chunks.append("\n\n".join(current))
                current, cur_tokens = [], 0
            chunks.extend(recursive_split_by_tokens(p, max_tokens, overlap_tokens))
            continue

        if cur_tokens + ptoks + 1 <= max_tokens:
            current.append(p)
            cur_tokens += ptoks + 1
        else:
            if current:
                chunks.append("\n\n".join(current))
                tail, tail_tokens = [], 0
                for para in reversed(current):
                    t = count_tokens(para)
                    tail.append(para)
                    tail_tokens += t
                    if tail_tokens >= overlap_tokens:  # Context-aware overlap
                        break
                current = list(reversed(tail))
                cur_tokens = sum(count_tokens(x) for x in current)
            current.append(p)
            cur_tokens += ptoks + 1
            
    if current:
        chunks.append("\n\n".join(current))

    final_chunks = []
    for ch in chunks:
        # --- DEFENSE-IN-DEPTH SANITIZATION ---
        # Sanitize the *final* chunk text before it's returned.
        # This catches any injections formed by joining paragraphs.
        if count_tokens(ch) > max_tokens:
            recursive_chunks = recursive_split_by_tokens(ch, max_tokens, overlap_tokens)
            final_chunks.extend([sanitize_for_injection(rc) for rc in recursive_chunks])
        else:
            final_chunks.append(sanitize_for_injection(ch))
            
    return final_chunks

def deduplicate(chunks, threshold=0.9, num_perm=128):
    logging.info(f"Deduplicating {len(chunks)} chunks with LSH (threshold={threshold})...")
    
    if not chunks:
        return []

    # Initialize LSH index
    lsh = MinHashLSH(threshold=threshold, num_perm=num_perm)
    
    # Create MinHashes for all chunks
    minhashes = {}
    for i, chunk in enumerate(chunks):
        key = f"chunk_{i}"
        
        # Tokenize the chunk to create a set of words
        tokens = set(tokenize(chunk.lower()))
        if not tokens:
            continue
            
        # Create a MinHash
        m = MinHash(num_perm=num_perm)
        for d in tokens:
            m.update(d.encode('utf8'))
        
        minhashes[key] = m
        lsh.insert(key, m)

    # Query to find duplicates and build the final list
    out = []
    seen_keys = set()

    for i, chunk in enumerate(chunks):
        key = f"chunk_{i}"
        
        # Skip if we don't have a hash or it's already part of a cluster
        if key not in minhashes or key in seen_keys:
            continue
            
        # Find near-duplicates in the LSH index
        result = lsh.query(minhashes[key])
        
        # Add the first item from this duplicate cluster (the current chunk)
        out.append(chunk)
        
        # Mark all members of this cluster as seen
        for res_key in result:
            seen_keys.add(res_key)
            
    logging.info(f"Reduced to {len(out)} chunks after LSH deduplication.")
    return out

def write_jsonl(path, rows):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        for r in rows:
            f.write(orjson.dumps(r, option=orjson.OPT_APPEND_NEWLINE))

def process_single_document(filepath, granularities, overlap_tokens, artifacts_dir, embedder=None):
    """Orchestrates the chunking of one document at multiple granularities."""
    fname = os.path.basename(filepath)
    doc_id = f"doc_{uuid.uuid4().hex[:8]}"
    logging.info(f"Parsing: {fname}")

    # Use unified loader for all file types
    cleaned = clean_and_parse_any(filepath)
    if not cleaned:
        logging.warning(f"No content extracted from {fname}, skipping.")
        return {"clean_chars": 0}

    stats = {"clean_chars": len(cleaned)}
    logging.info(f"Cleaned {fname}: {stats}")

    cleaned_out = os.path.join(artifacts_dir, "clean", f"{os.path.splitext(fname)[0]}__clean.txt")
    os.makedirs(os.path.dirname(cleaned_out), exist_ok=True)
    with open(cleaned_out, "w", encoding="utf-8") as f:
        f.write(cleaned)

    # We create one stable hash for the entire parent document
    parent_hash = hashlib.sha256(cleaned.encode("utf-8")).hexdigest()

    # Per-granularity overlap configuration with fallback
    overlap_config = {2048: 256, 512: 128, 128: 32}

    for g in granularities:  # Segmenting at multiple granularities
        # Use configured overlap for this granularity, fallback to parameter
        actual_overlap = overlap_config.get(g, overlap_tokens)
        logging.info(f"Chunking {fname} at granularity {g} with overlap {actual_overlap}")

        chunks = content_aware_chunk(cleaned, max_tokens=g, overlap_tokens=actual_overlap)
        # We don't need to re-normalize or re-sanitize here,
        # as content_aware_chunk now handles it.
        chunks = [c for c in chunks if count_tokens(c) >= 10]

        # Deduplication now uses LSH
        chunks = deduplicate(chunks, threshold=0.9)

        rows = []
        for i, ch in enumerate(chunks):
            row = {
                "chunk_id": f"{doc_id}_{g}_{i:05d}", "doc_id": doc_id,
                "parent_doc_hash": parent_hash, "source_path": os.path.abspath(filepath),
                "source_basename": fname, "granularity_tokens": g,
                "overlap_tokens": actual_overlap, "position": i,
                "num_positions": len(chunks), "text": ch, "n_tokens": count_tokens(ch),
                "prev_chunk_id": f"{doc_id}_{g}_{i-1:05d}" if i > 0 else None,
                "next_chunk_id": f"{doc_id}_{g}_{i+1:05d}" if i < len(chunks)-1 else None,
                "created_at": int(time.time()),
            }
            rows.append(row)

        out_path = os.path.join(artifacts_dir, "chunks", f"{g}_tokens_{os.path.splitext(fname)[0]}.jsonl")
        write_jsonl(out_path, rows)
        logging.info(f"Wrote {len(rows)} chunks to {out_path}")

        # Build embedding graph with optional shared embedder
        try:
            graph_dir = os.path.join(artifacts_dir, "graph")
            os.makedirs(graph_dir, exist_ok=True)
            save_document_graph(
                doc_id=doc_id,
                basename=fname,
                tokens=g,
                rows=rows,
                out_dir=graph_dir,
                # model="sentence-transformers/all-mpnet-base-v2",
                model=DEFAULT_EMBED_MODEL,
                embedder=embedder
            )
            logging.info(f"Wrote embedding graph for {fname} at {g} tokens → {graph_dir}")
        except Exception as e:
            logging.exception(f"Graph build failed for {fname} @ {g} tokens: {e}")
    return stats

def _json_lines(path):
    """Helper to read JSONL files."""
    try:
        with open(path, "rb") as f:
            for line in f:
                if line.strip():
                    yield orjson.loads(line)
    except Exception:
        import json
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    yield json.loads(line)

def run_batch(input_dir, artifacts_dir, granularities, overlap_tokens):
    os.makedirs(artifacts_dir, exist_ok=True)
    log_path = setup_logging(os.path.join(artifacts_dir, "logs"))
    logging.info("=== SME Preprocessing & Chunking Pipeline (Multi-Format, LSH Enabled, Sanitized) ===")
    logging.info(f"Input dir: {input_dir}")
    logging.info(f"Artifacts: {artifacts_dir}")
    logging.info(f"Granularities: {granularities}, overlap: {overlap_tokens}")

    # Use Part A to discover all supported file types recursively
    corpus, _ = collect_and_organize_documents(input_dir)

    # Flatten all file paths from corpus
    all_files = []
    for file_type, file_list in corpus.items():
        all_files.extend(file_list)

    if not all_files:
        logging.warning("No supported documents found. Exiting.")
        return

    logging.info(f"Found {len(all_files)} documents across {len(corpus)} file types")

    # Create shared TextEmbedder for efficiency
    logging.info("Initializing shared TextEmbedder...")
    # embedder = TextEmbedder(model="sentence-transformers/all-mpnet-base-v2")
    embedder = TextEmbedder(model=DEFAULT_EMBED_MODEL)


    # Process each document
    overall = {"files": 0, "clean_chars": 0, "errors": 0}
    for fp in tqdm(all_files, desc="Processing Documents", ncols=100):
        try:
            stats = process_single_document(fp, granularities, overlap_tokens, artifacts_dir, embedder=embedder)
            overall["files"] += 1
            overall["clean_chars"] += stats["clean_chars"]
        except Exception as e:
            overall["errors"] += 1
            logging.exception(f"Failed to process {fp}: {e}")

    logging.info(f"Per-document processing complete: {overall}")

    # Batch-level cross-document deduplication per granularity
    logging.info("=== Starting Batch-Level Cross-Document Deduplication ===")
    chunks_dir = os.path.join(artifacts_dir, "chunks")

    for g in granularities:
        try:
            logging.info(f"Batch dedup for granularity {g} tokens...")

            # Load all chunks for this granularity from all documents
            chunk_pattern = os.path.join(chunks_dir, f"{g}_tokens_*.jsonl")
            chunk_files = sorted(glob.glob(chunk_pattern))

            if not chunk_files:
                logging.warning(f"No chunk files found for {g} tokens, skipping batch dedup")
                continue

            all_chunks = []
            for cf in chunk_files:
                all_chunks.extend(list(_json_lines(cf)))

            logging.info(f"Loaded {len(all_chunks)} chunks from {len(chunk_files)} files @ {g} tokens")

            if not all_chunks:
                continue

            # Extract texts for deduplication
            texts = [c["text"] for c in all_chunks]

            # Deduplicate across documents using LSH
            deduped_texts = deduplicate(texts, threshold=0.9)
            deduped_set = set(deduped_texts)

            # Filter chunks to keep only deduplicated ones
            final_chunks = [c for c in all_chunks if c["text"] in deduped_set]

            # Write consolidated batch-deduplicated output
            batch_out_path = os.path.join(chunks_dir, f"{g}_tokens_batch_deduped.jsonl")
            write_jsonl(batch_out_path, final_chunks)

            logging.info(f"Batch dedup @ {g} tokens: {len(all_chunks)} → {len(final_chunks)} chunks")
            logging.info(f"Saved to: {batch_out_path}")

        except Exception as e:
            logging.exception(f"Batch deduplication failed for {g} tokens: {e}")

    logging.info(f"=== DONE | {overall} | Logs: {log_path}")