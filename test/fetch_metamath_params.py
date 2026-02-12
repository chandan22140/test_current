
import wandb
import pandas as pd
import sys

# Configuration
ENTITY = "chandan22140-indraprastha-institute-of-information-techn"
PROJECT = "Gemma SOARA"
METHOD = "way0"

api = wandb.Api()
results = []

print(f"Fetching results for {PROJECT} (method={METHOD})...", file=sys.stderr)

try:
    runs = api.runs(f"{ENTITY}/{PROJECT}")
    
    best_run = None
    best_acc = -float("inf")
    
    found_runs = 0
    for run in runs:
        # Filter by method if specified in config
        config_method = run.config.get("method")
        if config_method and config_method != METHOD:
            continue
            
        found_runs += 1
        
        # Check for accuracy metric
        # Common keys: 'eval/accuracy', 'accuracy', 'test_accuracy'
        # Also check for 'eval/loss' if accuracy not present (lower is better for loss)
        summary = run.summary
        
        acc = summary.get("eval/accuracy")
        if acc is None:
            acc = summary.get("accuracy")
        if acc is None:
            acc = summary.get("test_accuracy")
            
        if acc is not None:
            try:
                acc = float(acc)
                if acc > best_acc:
                    best_acc = acc
                    best_run = run
            except:
                continue
    
    print(f"Found {found_runs} runs matching method={METHOD}", file=sys.stderr)
    
    if best_run:
        print(f"Best Run found: {best_run.name} (Acc: {best_acc})", file=sys.stderr)
        cfg = best_run.config
        
        row = {
            "Dataset": "MetaMath (Gemma-7b)",
            "Variant": METHOD,
            "orthogonality_weight": cfg.get("orthogonality_weight") or cfg.get("ortho_weight") or 0.0, # default in script is 0
            "weight_decay": cfg.get("weight_decay") or 0.0, # default in script
            "learning_rate": cfg.get("learning_rate") or 2e-4, # default in script
            "rank": cfg.get("r") or cfg.get("lora_rank") or 128, # default in script
            "epoch": cfg.get("epochs") or cfg.get("num_train_epochs") or 1, # default in script
            "batch_size": cfg.get("s") or cfg.get("real_batch_size") or cfg.get("per_device_train_batch_size") or 128, # s=sample_size=128 default
            "lr_ratio_s": cfg.get("loraplus_lr_ratio"),
            "total_cycles": cfg.get("total_cycles"),
            "alpha": cfg.get("a") or cfg.get("lora_alpha") or 128,
            "test_accuracy": best_acc,
            "run_url": best_run.url
        }
        results.append(row)
    else:
        print("No run with accuracy metric found. Checking for best loss...", file=sys.stderr)
        # Fallback to loss if accuracy not found
        best_loss = float("inf")
        best_run_loss = None
        
        for run in runs:
            config_method = run.config.get("method")
            if config_method and config_method != METHOD:
                continue
                
            summary = run.summary
            loss = summary.get("eval/loss")
            if loss is None:
                loss = summary.get("loss")
            if loss is None:
                loss = summary.get("train/loss")

            if loss is not None:
                try:
                    loss = float(loss)
                    if loss < best_loss:
                        best_loss = loss
                        best_run_loss = run
                except:
                    continue
        
        if best_run_loss:
            print(f"Best Run (Loss) found: {best_run_loss.name} (Loss: {best_loss})", file=sys.stderr)
            cfg = best_run_loss.config
            row = {
                "Dataset": "MetaMath (Gemma-7b)",
                "Variant": METHOD,
                "orthogonality_weight": cfg.get("orthogonality_weight") or cfg.get("ortho_weight") or 0.0,
                "weight_decay": cfg.get("weight_decay") or 0.0,
                "learning_rate": cfg.get("learning_rate") or 2e-4,
                "rank": cfg.get("r") or cfg.get("lora_rank") or 128,
                "epoch": cfg.get("epochs") or cfg.get("num_train_epochs") or 1,
                "batch_size": cfg.get("s") or cfg.get("real_batch_size") or cfg.get("per_device_train_batch_size") or 128,
                "lr_ratio_s": cfg.get("loraplus_lr_ratio"),
                "total_cycles": cfg.get("total_cycles"),
                "alpha": cfg.get("a") or cfg.get("lora_alpha") or 128,
                "test_accuracy": f"Loss: {best_loss}", # Placeholder
                "run_url": best_run_loss.url
            }
            results.append(row)
        else:
            print("No runs found.", file=sys.stderr)
            # Add default values row
            print("Using default values from script...", file=sys.stderr)
            row = {
                "Dataset": "MetaMath (Gemma-7b) [DEFAULTS]",
                "Variant": METHOD,
                "orthogonality_weight": 0.0,
                "weight_decay": 0.0,
                "learning_rate": 2e-4,
                "rank": 128,
                "epoch": 1,
                "batch_size": 128,
                "lr_ratio_s": None,
                "total_cycles": 4, 
                "alpha": 128,
                "test_accuracy": "N/A",
                "run_url": "N/A"
            }
            results.append(row)

except Exception as e:
    print(f"Error accessing project: {e}", file=sys.stderr)

df = pd.DataFrame(results)
df.to_csv("metamath_params.csv", index=False)
print("Saved to metamath_params.csv")
print(df)
