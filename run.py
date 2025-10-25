from part_B_preprocessing import run_batch
def main():
    input_dir = "./data"           # PDFs here
    artifacts_dir = "./artifacts"  # outputs (clean/chunks/graph/logs)
    granularities = [2048, 512, 128]
    overlap_tokens = 64

    run_batch(
        input_dir=input_dir,
        artifacts_dir=artifacts_dir,
        granularities=granularities,
        overlap_tokens=overlap_tokens,
    )
if __name__ == '__main__':
    main()