
import wandb
import pandas as pd
import sys

# Configuration
ENTITY = "chandan22140-indraprastha-institute-of-information-techn"
GLUE_TASKS = [
    "mrpc", "qnli", "rte", "cola", "stsb", "sst2", "mnli", "qqp"
]

# Metric mapping based on train_glue_deberta.py logs
# Keys are checked in order
METRIC_MAPPING = {
    "cola": ["val_matthews", "eval/matthews", "matthews"],
    "stsb": ["val_corr", "eval/corr", "val_spearman", "eval/spearman"],
    "mrpc": ["val_acc_f1", "eval/acc_f1", "val_f1", "eval/f1", "val_accuracy"],
    "qqp": ["val_acc_f1", "eval/acc_f1", "val_f1", "eval/f1", "val_accuracy"],
    "mnli": ["val_accuracy", "eval/accuracy", "accuracy"], # Check if val_m_mm exists but accuracy is standard
    "sst2": ["val_accuracy", "eval/accuracy", "accuracy"],
    "qnli": ["val_accuracy", "eval/accuracy", "accuracy"],
    "rte":  ["val_accuracy", "eval/accuracy", "accuracy"],
}

api = wandb.Api()
results = []

def get_best_run_from_list(runs, metric_keys, method_filter=None):
    best_run = None
    best_val = -float("inf")
    
    for run in runs:
        if run.state != "finished":
            continue
            
        # Filter by method if specified
        if method_filter:
            config_method = run.config.get("method")
            if config_method != method_filter:
                continue

        summary = run.summary
        metric_val = None
        
        # Check specific metric keys first
        for key in metric_keys:
            if key in summary:
                metric_val = summary[key]
                break
        
        # Fallback to 'best_metric' if available
        if metric_val is None:
            metric_val = summary.get("best_metric")
            
        if metric_val is None:
            continue
            
        try:
            val = float(metric_val)
            if val > best_val:
                best_val = val
                best_run = run
        except:
            continue
            
    return best_run, best_val

print("Fetching NLU/NLG results...", file=sys.stderr)

for task in GLUE_TASKS:
    print(f"--- Processing {task.upper()} ---", file=sys.stderr)
    metrics = METRIC_MAPPING.get(task, ["val_accuracy", "accuracy"])
    
    
    # Base params template
    base_params = {
        "Dataset": task.upper(),
        "Variant": None,
        "orthogonality_weight": None,
        "weight_decay": None,
        "learning_rate": None,
        "rank": None,
        "epoch": None,
        "batch_size": None,
        "lr_ratio_s": None,
        "total_cycles": None,
        "test_metric": None,
        "run_url": None
    }

    # 1. Way 1 BF Seq
    project_bf = f"glue-deberta-{task}-butterfly-sequential"
    params_bf = base_params.copy()
    params_bf["Variant"] = "way1 BF Seq"
    
    try:
        runs = api.runs(f"{ENTITY}/{project_bf}")
        best_run, best_val = get_best_run_from_list(runs, metrics) 
        
        if best_run:
            cfg = best_run.config
            params_bf.update({
                "orthogonality_weight": cfg.get("orthogonality_weight"),
                "weight_decay": cfg.get("weight_decay"),
                "learning_rate": cfg.get("learning_rate"),
                "rank": cfg.get("pissa_rank") or cfg.get("rank"),
                "epoch": cfg.get("epochs"),
                "batch_size": cfg.get("batch_size"),
                "lr_ratio_s": cfg.get("lr_ratio_s"),
                "total_cycles": cfg.get("total_cycles"),
                "test_metric": best_val,
                "run_url": best_run.url
            })
            print(f"  Way 1 BF Seq: {best_val}", file=sys.stderr)
        else:
            print(f"  Way 1 BF Seq: No run found", file=sys.stderr)
            
        results.append(params_bf)
    except Exception as e:
        print(f"  Way 1 BF Seq Error: {e}", file=sys.stderr)
        results.append(params_bf)


    # Mixed Project: glue-deberta-{task}
    project_mixed = f"glue-deberta-{task}"
    try:
        runs = api.runs(f"{ENTITY}/{project_mixed}")
        
        # 2. Way 0
        params_w0 = base_params.copy()
        params_w0["Variant"] = "way0"
        best_run_w0, best_val_w0 = get_best_run_from_list(runs, metrics, method_filter="way0")
        
        if best_run_w0:
            cfg = best_run_w0.config
            params_w0.update({
                "orthogonality_weight": cfg.get("orthogonality_weight"),
                "weight_decay": cfg.get("weight_decay"),
                "learning_rate": cfg.get("learning_rate"),
                "rank": cfg.get("pissa_rank") or cfg.get("rank"),
                "epoch": cfg.get("epochs"),
                "batch_size": cfg.get("batch_size"),
                "lr_ratio_s": cfg.get("lr_ratio_s"),
                "total_cycles": cfg.get("total_cycles"),
                "test_metric": best_val_w0,
                "run_url": best_run_w0.url
            })
            print(f"  Way 0: {best_val_w0}", file=sys.stderr)
        else:
             print(f"  Way 0: No run found", file=sys.stderr)
        results.append(params_w0)
        
        # 3. Way 1 (Standard)
        params_w1 = base_params.copy()
        params_w1["Variant"] = "way1"
        best_run_w1, best_val_w1 = get_best_run_from_list(runs, metrics, method_filter="way1")
        
        if best_run_w1:
            cfg = best_run_w1.config
            params_w1.update({
                "orthogonality_weight": cfg.get("orthogonality_weight"),
                "weight_decay": cfg.get("weight_decay"),
                "learning_rate": cfg.get("learning_rate"),
                "rank": cfg.get("pissa_rank") or cfg.get("rank"),
                "epoch": cfg.get("epochs"),
                "batch_size": cfg.get("batch_size"),
                "lr_ratio_s": cfg.get("lr_ratio_s"),
                "total_cycles": cfg.get("total_cycles"),
                "test_metric": best_val_w1,
                "run_url": best_run_w1.url
            })
            print(f"  Way 1: {best_val_w1}", file=sys.stderr)
        else:
             print(f"  Way 1: No run found", file=sys.stderr)
        results.append(params_w1)

    except Exception as e:
        print(f"  Mixed Project Error: {e}", file=sys.stderr)
        # Add empty rows for w0/w1 if project access fails
        params_w0 = base_params.copy()
        params_w0["Variant"] = "way0"
        results.append(params_w0)
        params_w1 = base_params.copy()
        params_w1["Variant"] = "way1"
        results.append(params_w1)

df = pd.DataFrame(results)
# Order columns
cols = ["Dataset", "Variant", "orthogonality_weight", "weight_decay", "learning_rate", 
        "rank", "epoch", "batch_size", "lr_ratio_s", "total_cycles", "test_metric", "run_url"]
df = df[cols]

df.to_csv("nlu_sweep_results.csv", index=False)
print("Saved to nlu_sweep_results.csv")
print(df)
