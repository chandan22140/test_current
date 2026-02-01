import os
import sys
import torchvision
import torchvision.transforms as transforms
from datasets import load_dataset
from huggingface_hub import hf_hub_download
import pickle

# Define data path
DATA_PATH = "./data"
os.makedirs(DATA_PATH, exist_ok=True)

print(f"📂 Downloading datasets to: {os.path.abspath(DATA_PATH)}")

def download_cifar100():
    print("\n⬇️  Downloading CIFAR-100...")
    try:
        torchvision.datasets.CIFAR100(root=DATA_PATH, train=True, download=True)
        torchvision.datasets.CIFAR100(root=DATA_PATH, train=False, download=True)
        print("✅ CIFAR-100 downloaded successfully.")
    except Exception as e:
        print(f"❌ Failed to download CIFAR-100: {e}")

def download_dtd():
    print("\n⬇️  Downloading DTD...")
    try:
        torchvision.datasets.DTD(root=DATA_PATH, split='train', download=True)
        torchvision.datasets.DTD(root=DATA_PATH, split='val', download=True)
        torchvision.datasets.DTD(root=DATA_PATH, split='test', download=True)
        print("✅ DTD downloaded successfully.")
    except Exception as e:
        print(f"❌ Failed to download DTD: {e}")

def download_fer2013():
    print("\n⬇️  Downloading FER2013...")
    try:
        train_pt = hf_hub_download(repo_id="Jeneral/fer-2013", filename="train.pt", 
                                   repo_type="dataset", cache_dir=DATA_PATH)
        test_pt = hf_hub_download(repo_id="Jeneral/fer-2013", filename="test.pt",
                                  repo_type="dataset", cache_dir=DATA_PATH)
        print("✅ FER2013 downloaded successfully.")
    except Exception as e:
        print(f"❌ Failed to download FER2013: {e}")

def download_fgvc_aircraft():
    print("\n⬇️  Downloading FGVC-Aircraft...")
    try:
        torchvision.datasets.FGVCAircraft(root=DATA_PATH, split='train', download=True)
        torchvision.datasets.FGVCAircraft(root=DATA_PATH, split='val', download=True)
        torchvision.datasets.FGVCAircraft(root=DATA_PATH, split='test', download=True)
        print("✅ FGVC-Aircraft downloaded successfully.")
    except Exception as e:
        print(f"❌ Failed to download FGVC-Aircraft: {e}")

def download_sun397():
    print("\n⬇️  Downloading SUN397...")
    try:
        # Using Hugging Face as per main script
        load_dataset("tanganke/sun397", cache_dir=DATA_PATH)
        print("✅ SUN397 downloaded successfully.")
    except Exception as e:
        print(f"❌ Failed to download SUN397: {e}")

if __name__ == "__main__":
    download_cifar100()
    download_dtd()
    download_fer2013()
    download_fgvc_aircraft()
    download_sun397()
    
    print("\n✨ All download tasks completed.")
