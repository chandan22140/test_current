
import wandb
import pandas as pd
import sys

# Define the tasks
# Format: (Dataset, Variant, Entity, Project, SweepID_or_None)
tasks = [
    # Way 1, BF seq
    ("CIFAR100", "SOARA (way1, BF seq)", "chandan22140-indraprastha-institute-of-information-techn", "PiSSA-cifar100-butterfly-seq", "wsnfh5ap"),
    ("DTD", "SOARA (way1, BF seq)", "chandan22140-indraprastha-institute-of-information-techn", "PiSSA-dtd-butterfly-seq", "tds0t3og"),
    ("SUN397", "SOARA (way1, BF seq)", "chandan22140-indraprastha-institute-of-information-techn", "PiSSA-sun397-butterfly-seq", "7ly1thqs"),
    ("FER2013", "SOARA (way1, BF seq)", "chandan22140-indraprastha-institute-of-information-techn", "PiSSA-fer-butterfly-seq", "byd8yxla"),
    ("FGVCAircraft", "SOARA (way1, BF seq)", "chandan22140-indraprastha-institute-of-information-techn", "PiSSA-fgvc-butterfly-seq", "bb991a5d"),
    ("Food101", "SOARA (way1, BF seq)", "chandan22140-indraprastha-institute-of-information-techn", "PiSSA-food101-butterfly-seq", "337mcrru"),
    ("Resisc45", "SOARA (way1, BF seq)", "chandan22140-indraprastha-institute-of-information-techn", "PiSSA-resisc45-butterfly-seq", "ce1sltgo"),
    
    # Way 0, r=16
    ("CIFAR100", "SOARA (way0, r=16)", "chandan22140-indraprastha-institute-of-information-techn", "rotational-pissa-cifar100", "dkmgosbl"),
    ("Food101", "SOARA (way0, r=16)", "chandan22140-indraprastha-institute-of-information-techn", "rotational-pissa-food101", "h59nybto"),
    ("Resisc45", "SOARA (way0, r=16)", "chandan22140-indraprastha-institute-of-information-techn", "rotational-pissa-resisc45", "sxvy6m9p"),
    ("DTD", "SOARA (way0, r=16)", "chandan22140-indraprastha-institute-of-information-techn", "rotational-pissa-dtd", None),
    ("SUN397", "SOARA (way0, r=16)", "chandan22140-indraprastha-institute-of-information-techn", "rotational-pissa-sun397", None),
    ("FER2013", "SOARA (way0, r=16)", "chandan22140-indraprastha-institute-of-information-techn", "rotational-pissa-fer2013", None),
    ("FGVCAircraft", "SOARA (way0, r=16)", "chandan22140-indraprastha-institute-of-information-techn", "rotational-pissa-fgvc_aircraft", None),

    # Way 1, Givens, r=16
    ("CIFAR100", "SOARA (way1, givens, r=16)", "chandan22140-indraprastha-institute-of-information-techn", "PiSSA-cifar100-way1", "jgytzgha"),
    ("DTD", "SOARA (way1, givens, r=16)", "chandan22140-indraprastha-institute-of-information-techn", "PiSSA-dtd-way1", "rdvsr3rz"),
    ("FER2013", "SOARA (way1, givens, r=16)", "chandan22140-indraprastha-institute-of-information-techn", "PiSSA-fer-way1", "r0xnfhu3"),
    ("FGVCAircraft", "SOARA (way1, givens, r=16)", "chandan22140-indraprastha-institute-of-information-techn", "PiSSA-fgvc-way1", "d0s9r55n"),
    ("SUN397", "SOARA (way1, givens, r=16)", "chandan22140-indraprastha-institute-of-information-techn", "PiSSA-sun397-way1", "mnymndqy"),
    ("Food101", "SOARA (way1, givens, r=16)", "chandan22140-indraprastha-institute-of-information-techn", "PiSSA-food101-way1", "xthuqlse"),
    ("Resisc45", "SOARA (way1, givens, r=16)", "chandan22140-indraprastha-institute-of-information-techn", "PiSSA-resisc45-way1", "ajzixs4w"),
]

api = wandb.Api()
results = []

def get_best_run(runs):
    best_run = None
    best_acc = -1.0
    for run in runs:
        if run.state != "finished":
            continue
        s = run.summary
        acc = s.get("test_accuracy")
        if acc is None:
            acc = s.get("accuracy")
        if acc is None:
            acc = s.get("eval/accuracy")
        
        if acc is not None:
            try:
                acc = float(acc)
                if acc > best_acc:
                    best_acc = acc
                    best_run = run
            except:
                continue
    return best_run, best_acc

print("Fetching results...", file=sys.stderr)

for dataset, variant, entity, project, sweep_id in tasks:
    print(f"Processing {dataset} - {variant}...", file=sys.stderr)
    try:
        runs = None
        sweep_url = "N/A"
        
        if sweep_id:
            try:
                sweep_path = f"{entity}/{project}/{sweep_id}"
                sweep = api.sweep(sweep_path)
                runs = sweep.runs
                sweep_url = sweep.url
            except Exception as e:
                print(f"  Could not access sweep {sweep_id}: {e}", file=sys.stderr)
        
        if runs is None or len(runs) == 0:
            # Try finding a sweep in the project
            print(f"  Searching for sweeps/runs in project {project}...", file=sys.stderr)
            try:
                sweeps = api.project(project, entity=entity).sweeps()
                if sweeps and len(sweeps) > 0:
                    sweep = sweeps[0] # Use the first one (usually most recent)
                    print(f"  Found sweep {sweep.id}", file=sys.stderr)
                    runs = sweep.runs
                    sweep_url = sweep.url
                else:
                    # Look for runs directly in project
                    print(f"  No sweeps found, checking runs directly...", file=sys.stderr)
                    runs = api.runs(f"{entity}/{project}")
                    sweep_url = f"https://wandb.ai/{entity}/{project}"
            except Exception as e:
                 print(f"  Error accessing project: {e}", file=sys.stderr)
                 continue

        if runs:
            best_run, best_acc = get_best_run(runs)
            
            if best_run:
                cfg = best_run.config
                row = {
                    "Dataset": dataset,
                    "Variant": variant,
                    "orthogonality_weight": cfg.get("orthogonality_weight"),
                    "weight_decay": cfg.get("weight_decay"),
                    "learning_rate": cfg.get("learning_rate"),
                    "rank": cfg.get("pissa_rank") or cfg.get("rank") or cfg.get("r"),
                    "epoch": cfg.get("epochs") or cfg.get("num_train_epochs"),
                    "batch_size": cfg.get("batch_size") or cfg.get("per_device_train_batch_size"),
                    "lr-ratio-s": cfg.get("lr_ratio_s") or cfg.get("lr_ratio"),
                    "total-cycles": cfg.get("total_cycles") or cfg.get("num_cycles") or cfg.get("cycles"),
                    "test_accuracy": best_acc,
                    "sweep_url": sweep_url
                }
                results.append(row)
                print(f"  Best Acc: {best_acc}", file=sys.stderr)
            else:
                print(f"  No valid runs found with accuracy metric.", file=sys.stderr)
        else:
             print(f"  No runs found.", file=sys.stderr)

    except Exception as e:
        print(f"  Error: {e}", file=sys.stderr)

# Create DataFrame directly with specific column order
columns = [
    "Dataset", "Variant", 
    "orthogonality_weight", "weight_decay", "learning_rate", 
    "rank", "epoch", "batch_size", "lr-ratio-s", "total-cycles", 
    "test_accuracy", "sweep_url"
]
df = pd.DataFrame(results, columns=columns)
df.to_csv("sweep_results.csv", index=False)
print("Saved to sweep_results.csv")
print(df)
