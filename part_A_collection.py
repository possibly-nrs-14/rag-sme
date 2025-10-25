import os
import glob
import logging
import sys
from collections import defaultdict
import json 

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)

def collect_and_organize_documents(input_dir):
    logging.info(f"Scanning for documents in: {input_dir}")
    
    supported_extensions = ["*.pdf", "*.docx", "*.pptx", "*.txt", "*.md"]
    
    all_files = []
    for ext in supported_extensions:
        files = glob.glob(os.path.join(input_dir, "**", ext), recursive=True)
        all_files.extend(files)
        
    logging.info(f"Found {len(all_files)} total files.")
    organized_corpus = defaultdict(list)
    metadata = {}
    for fp in all_files:
        file_name = os.path.basename(fp)
        file_ext = os.path.splitext(fp)[1].lower()
        abs_path = os.path.abspath(fp)
        organized_corpus[file_ext].append(abs_path)
        metadata[abs_path] = {
            "file_name": file_name,
            "file_type": file_ext,
        }

    for k in organized_corpus:
        organized_corpus[k] = sorted(organized_corpus[k])
        
    return organized_corpus, metadata

def main():
    INPUT_DIR = "./data"
    corpus, metadata = collect_and_organize_documents(INPUT_DIR)
    logging.info(f"Total metadata entries: {len(metadata)}")
    os.makedirs("./metadata", exist_ok=True)
    metadata_path = "./metadata/file_metadata.json"
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
    logging.info(f"Metadata summary saved to {metadata_path}")

    print("\nCorpus organized by file type:")
    print("--------------------------------")
    if not corpus:
        print("No documents found.")
        return

    for file_type, files in corpus.items():
        print(f"Found {len(files)} files for type: {file_type}")
        for f in files[:5]:
            print(f"  - {f}")

if __name__ == "__main__":
    main()
