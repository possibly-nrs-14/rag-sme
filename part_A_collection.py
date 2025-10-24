# part_A_collection.py
import os
import glob
import logging
import sys
from collections import defaultdict

# Set up a simple logger for this part
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)

def collect_and_organize_documents(input_dir):
    """
    Collects documents from a directory and organizes them by file type.
    This fulfills the requirements of Part A[cite: 26].
    """
    logging.info(f"Scanning for documents in: {input_dir}")
    
    # The project must support heterogeneous formats 
    supported_extensions = ["*.pdf", "*.docx", "*.pptx", "*.txt", "*.md"]
    
    all_files = []
    for ext in supported_extensions:
        # Use recursive=True to find files in subdirectories
        files = glob.glob(os.path.join(input_dir, "**", ext), recursive=True)
        all_files.extend(files)
        
    logging.info(f"Found {len(all_files)} total files.")
    
    # Automatically detect types and process documents [cite: 30]
    organized_corpus = defaultdict(list)
    for fp in all_files:
        file_ext = os.path.splitext(fp)[1].lower()
        organized_corpus[file_ext].append(os.path.abspath(fp))
        
    # Metadata association is also a requirement[cite: 29]. This could be expanded
    # by parsing filenames or reading a master JSON file.
    
    return organized_corpus

def main():
    """Main execution block."""
    INPUT_DIR = "./data"
    corpus = collect_and_organize_documents(INPUT_DIR)

    print("\nCorpus organized by file type:")
    print("--------------------------------")
    if not corpus:
        print("No documents found.")
        return

    for file_type, files in corpus.items():
        print(f"Found {len(files)} files for type: {file_type}")
        # Print first 5 as a sample
        for f in files[:5]:
            print(f"  - {f}")

if __name__ == "__main__":
    main()
