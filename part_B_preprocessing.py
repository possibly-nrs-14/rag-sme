# part_B_preprocessing.py
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
import fitz  # PyMuPDF
from datasketch import MinHash, MinHashLSH  # <-- Added this import

# Regex for tokenization, needed by LSH function
_WORD_RE = re.compile(r"\w+|[^\w\s]", re.UNICODE)

def setup_logging(log_dir):
    """Sets up comprehensive error logging."""
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
    """Part B: Tokenization (used by chunking and LSH)."""
    return _WORD_RE.findall(s)

def count_tokens(s):
    return len(tokenize(s))

def normalize_spaces(s):
    """Part B: Preprocessing including lowercasing and content removal."""
    s = re.sub(r"[ \t]+", " ", s)
    s = re.sub(r"\u00A0", " ", s)
    s = re.sub(r" *\n *", "\n", s)
    s = re.sub(r"\n{3,}", "\n\n", s)
    return s.strip()

def clean_pdf(path, threshold=10):
    """Parses and cleans a single PDF, removing non-informative content."""
    doc = fitz.open(path)
    base = os.path.basename(path)
    page_texts = []
    book_start = False
    book_end = False

    for i in range(len(doc)):
        try:
            page = doc[i]
            blocks = page.get_text("blocks")
            if not blocks:
                page_texts.append("")
                continue

            # Your existing logic for column detection and ordering
            mid_x = (page.rect.x0 + page.rect.x1) / 2
            left_blocks, right_blocks = [], []
            for b in blocks:
                if len(b) < 5: continue
                x0, y0, x1, y1, txt = b[0], b[1], b[2], b[3], b[4] or ""
                if not txt.strip(): continue
                cx = (x0 + x1) / 2
                (left_blocks if cx < mid_x else right_blocks).append((y0, x0, txt))

            left_blocks.sort(key=lambda t: (t[0], t[1]))
            right_blocks.sort(key=lambda t: (t[0], t[1]))
            ordered = left_blocks + right_blocks

            kept_lines = []
            for j, (_, _, txt) in enumerate(ordered):
                plain = normalize_spaces(txt).lower()  # Lowercasing
                n_words = len(plain.split())
                
                if not book_start and ("c h a p t e r" in plain or "gross anatomy of the brain" in plain):
                    book_start = True
                if plain in ["index", "i n d e x"] and (j == 0 or len(doc) - i <= 20):
                    book_end = True
                    break
                if not book_start:
                    continue
                if n_words < threshold:  # Removal of non-informative content
                    continue
                kept_lines.append(plain)

            page_texts.append("\n".join(kept_lines))
        except Exception as e:
            logging.exception(f"Failed to process blocks on page {i+1} of {base}: {e}")
            page_texts.append("")
        if book_end:
            break
            
    doc.close()
    return page_texts

def clean_and_parse_pdf(path):
    pages = clean_pdf(path)
    return "\n\n".join(pages)

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
        if count_tokens(ch) > max_tokens:
            final_chunks.extend(recursive_split_by_tokens(ch, max_tokens, overlap_tokens))
        else:
            final_chunks.append(ch)
    return final_chunks

# --- REPLACED DEDUPLICATION SECTION ---

def deduplicate(chunks, threshold=0.9, num_perm=128):
    """
    Deduplicates a list of text chunks using MinHash LSH.
    Keeps the first-encountered chunk of any near-duplicate set.
    
    threshold: Jaccard similarity threshold (0.0 to 1.0).
               0.9 means 90% similar.
    num_perm:  Number of permutations, affects signature accuracy.
               128 is a good default.
    """
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
# --- END REPLACED SECTION ---


def write_jsonl(path, rows):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        for r in rows:
            f.write(orjson.dumps(r, option=orjson.OPT_APPEND_NEWLINE))

def process_single_pdf(filepath, granularities, overlap_tokens, artifacts_dir):
    """Orchestrates the chunking of one PDF at multiple granularities."""
    fname = os.path.basename(filepath)
    doc_id = f"doc_{uuid.uuid4().hex[:8]}"
    logging.info(f"Parsing: {fname}")

    cleaned = clean_and_parse_pdf(filepath)
    stats = {"clean_chars": len(cleaned)}
    logging.info(f"Cleaned {fname}: {stats}")

    cleaned_out = os.path.join(artifacts_dir, "clean", f"{os.path.splitext(fname)[0]}__clean.txt")
    os.makedirs(os.path.dirname(cleaned_out), exist_ok=True)
    with open(cleaned_out, "w", encoding="utf-8") as f:
        f.write(cleaned)
    
    # We create one stable hash for the entire parent document
    parent_hash = hashlib.sha256(cleaned.encode("utf-8")).hexdigest()

    for g in granularities:  # Segmenting at multiple granularities
        logging.info(f"Chunking {fname} at granularity {g} with overlap {overlap_tokens}")
        chunks = content_aware_chunk(cleaned, max_tokens=g, overlap_tokens=overlap_tokens)
        chunks = [normalize_spaces(c) for c in chunks]
        chunks = [c for c in chunks if count_tokens(c) >= 10]
        
        # Deduplication now uses LSH
        chunks = deduplicate(chunks, threshold=0.9) 

        rows = []
        for i, ch in enumerate(chunks):
            row = {
                "chunk_id": f"{doc_id}_{g}_{i:05d}", "doc_id": doc_id,
                "parent_doc_hash": parent_hash, "source_path": os.path.abspath(filepath),
                "source_basename": fname, "granularity_tokens": g,
                "overlap_tokens": overlap_tokens, "position": i,
                "num_positions": len(chunks), "text": ch, "n_tokens": count_tokens(ch),
                "prev_chunk_id": f"{doc_id}_{g}_{i-1:05d}" if i > 0 else None,
                "next_chunk_id": f"{doc_id}_{g}_{i+1:05d}" if i < len(chunks)-1 else None,
                "created_at": int(time.time()),
            }
            rows.append(row)

        out_path = os.path.join(artifacts_dir, "chunks", f"{g}_tokens_{os.path.splitext(fname)[0]}.jsonl")
        write_jsonl(out_path, rows)
        logging.info(f"Wrote {len(rows)} chunks to {out_path}")
    return stats

def run_batch(input_dir, artifacts_dir, granularities, overlap_tokens):
    """BONUS: The automated batch ingestion pipeline."""
    os.makedirs(artifacts_dir, exist_ok=True)
    log_path = setup_logging(os.path.join(artifacts_dir, "logs"))
    logging.info("=== SME Preprocessing & Chunking Pipeline (LSH Enabled) ===")
    logging.info(f"Input dir: {input_dir}")
    logging.info(f"Artifacts: {artifacts_dir}")
    logging.info(f"Granularities: {granularities}, overlap: {overlap_tokens}")
    
    pdfs = sorted(glob.glob(os.path.join(input_dir, "*.pdf")))
    if not pdfs:
        logging.warning("No PDFs found. Exiting.")
        return

    overall = {"files": 0, "clean_chars": 0, "errors": 0}
    for fp in tqdm(pdfs, desc="Processing PDFs", ncols=100):
        try:
            stats = process_single_pdf(fp, granularities, overlap_tokens, artifacts_dir)
            overall["files"] += 1
            overall["clean_chars"] += stats["clean_chars"]
        except Exception as e:
            overall["errors"] += 1
            logging.exception(f"Failed to process {fp}: {e}")

    logging.info(f"=== DONE | {overall} | Logs: {log_path}")

def main():
    """Main execution block."""
    run_batch(
        input_dir="./data", 
        artifacts_dir="./artifacts", 
        granularities=[2048, 512, 128], # As required
        overlap_tokens=64
    )

if __name__ == "__main__":
    main()
