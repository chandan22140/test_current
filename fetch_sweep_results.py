import wandb
import pandas as pd

# List of sweeps URL components or full paths
sweeps = [
    {"name": "Butterfly Sequential - Food101", "path": "chandan22140-indraprastha-institute-of-information-techn/PiSSA-food101-butterfly-seq/sweeps/337mcrru"},
    {"name": "Butterfly Sequential - Resisc45", "path": "chandan22140-indraprastha-institute-of-information-techn/PiSSA-resisc45-butterfly-seq/sweeps/ce1sltgo"},
    {"name": "Way 1 (Rank 16) - Food101", "path": "chandan22140-indraprastha-institute-of-information-techn/PiSSA-food101-way1/sweeps/xthuqlse"},
    {"name": "Way 1 (Rank 16) - Resisc45", "path": "chandan22140-indraprastha-institute-of-information-techn/PiSSA-resisc45-way1/sweeps/ajzixs4w"},
    {"name": "Way 0 (Rank 16) - Food101", "path": "chandan22140-indraprastha-institute-of-information-techn/rotational-pissa-food101/sweeps/h59nybto"},
    {"name": "Way 0 (Rank 16) - Resisc45", "path": "chandan22140-indraprastha-institute-of-information-techn/rotational-pissa-resisc45/sweeps/sxvy6m9p"},
]

wandb.login(key="5b72e552516fbddcc3131462654d458952315d26")
api = wandb.Api()

print("Fetching sweep results...\n")

for sweep_info in sweeps:
    print(f"--- {sweep_info['name']} ---")
    try:
        sweep = api.sweep(sweep_info['path'])
        
        # Get all runs in the sweep
        runs = sweep.runs
        
        best_run = None
        best_accuracy = -1.0
        
        for run in runs:
            # Check summary metrics for accuracy
            # Keys might differ slightly, usually 'test_accuracy' or 'acc', 'val_acc'
            # We look for 'test_accuracy' as requested, but fall back to common ones if missing
            metrics = run.summary
            
            acc = metrics.get('test_accuracy')
            if acc is None:
                acc = metrics.get('accuracy')
            if acc is None:
                acc = metrics.get('eval/accuracy')
                
            if acc is not None:
                # Ensure it's a float
                try:
                    acc_val = float(acc)
                    if acc_val > best_accuracy:
                        best_accuracy = acc_val
                        best_run = run
                except:
                    continue
        
        if best_run:
            print(f"Best Test Accuracy: {best_accuracy}")
            print("Hyperparameters:")
            # Filter out wandb internal config
            for k, v in best_run.config.items():
                if not k.startswith('_') and not k.startswith('wandb'):
                    print(f"  {k}: {v}")
            print(f"Run URL: {best_run.url}")
        else:
            print("No runs found with 'test_accuracy' (or 'accuracy'/'eval/accuracy') metric.")
            
    except Exception as e:
        print(f"Error fetching sweep {sweep_info['path']}: {e}")
    
    print("\n")
