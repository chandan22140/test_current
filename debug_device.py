
from accelerate import Accelerator
import torch
import os
import time

def main():
    accelerator = Accelerator()
    print(f"Process {os.getpid()} | Rank {accelerator.process_index} | Local Rank {accelerator.local_process_index}")
    print(f"Process {os.getpid()} | Accelerator Device: {accelerator.device}")
    
    # Check what torch thinks
    if torch.cuda.is_available():
        print(f"Process {os.getpid()} | CUDA Available. Device Count: {torch.cuda.device_count()}")
        print(f"Process {os.getpid()} | Current Device Index: {torch.cuda.current_device()}")
        print(f"Process {os.getpid()} | Current Device Name: {torch.cuda.get_device_name(torch.cuda.current_device())}")
    
    # Check env vars
    print(f"Process {os.getpid()} | CUDA_VISIBLE_DEVICES: {os.environ.get('CUDA_VISIBLE_DEVICES')}")

    # Allocate a small tensor on 'current device'
    try:
        t = torch.tensor([1.0]).cuda()
        print(f"Process {os.getpid()} | Allocated tensor on: {t.device}")
    except Exception as e:
        print(f"Process {os.getpid()} | Allocation failed: {e}")

if __name__ == "__main__":
    main()
