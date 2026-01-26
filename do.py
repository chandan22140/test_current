import datasets
from datasets import load_dataset

def download_missing_datasets():
    # RTE and MRPC were reported missing from cache
    tasks = ['rte', 'mrpc']
    
    print(f"Attempting to download missing GLUE tasks: {tasks}")
    
    for task in tasks:
        try:
            print(f"\nDownloading {task}...")
            # Force download mode if needed, but standard load should cache it
            dataset = load_dataset("glue", task, download_mode="force_redownload")
            print(f"✅ Successfully cached {task}")
            print(f"  Train: {len(dataset['train'])} samples")
            print(f"  Validation: {len(dataset['validation'])} samples")
        except Exception as e:
            print(f"❌ Failed to download {task}: {e}")

if __name__ == "__main__":
    download_missing_datasets()
