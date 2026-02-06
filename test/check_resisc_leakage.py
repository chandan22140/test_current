
import os
import sys
from datasets import load_dataset
import numpy as np

def check_resisc():
    print("Loading timm/resisc45...")
    try:
        # Load the dataset (stats only mode if possible, but load_dataset usually downloads)
        # We assume it's already cached since the user ran it
        ds = load_dataset("timm/resisc45")
        
        print("Keys found:", ds.keys())
        
        splits = list(ds.keys())
        for split in splits:
            print(f"Split '{split}': {len(ds[split])} samples")
            
        # Check for overlap
        if len(splits) > 1:
            print("\nChecking for filename/ID overlap...")
            # Collect filenames if available
            # We look at the first item to see structure
            sample = ds[splits[0]][0]
            print("Sample keys:", sample.keys())
            
            # If 'file_name' or similar exists, use it. 
            # Otherwise we might need to rely on exact image bytes matching (slower) or hashing
            
            overlap_found = False
            
            # Helper to get identifiers
            def get_ids(split_name):
                d = ds[split_name]
                ids = []
                # Try common ID keys
                keys_to_try = ['file_name', 'filename', 'id', 'image_id']
                target_key = None
                for k in keys_to_try:
                    if k in d.features:
                        target_key = k
                        break
                
                if target_key:
                    print(f"Using '{target_key}' as identifier for {split_name}")
                    return set(d[target_key])
                else:
                    print(f"No filename/id found for {split_name}. Using (label, index) is not sufficient for overlap check.")
                    return None

            split_ids = {}
            for split in splits:
                ident = get_ids(split)
                if ident:
                    split_ids[split] = ident
            
            # Compare sets
            import itertools
            for s1, s2 in itertools.combinations(split_ids.keys(), 2):
                intersection = split_ids[s1].intersection(split_ids[s2])
                if intersection:
                    print(f"WARNING: Found {len(intersection)} overlapping items between {s1} and {s2}!")
                    print(f"Example overlap: {list(intersection)[:5]}")
                    overlap_found = True
                else:
                    print(f"No overlap found between {s1} and {s2} (based on identifiers).")
            
            if not split_ids:
                print("Could not find identifiers. Checking first 100 hashes...")
                # Fallback: hash images
                import hashlib
                
                def get_hashes(split_name, limit=1000):
                    hashes = set()
                    d = ds[split_name]
                    n = min(len(d), limit)
                    for i in range(n):
                        img = d[i]['image']
                        # Convert to bytes
                        if hasattr(img, 'tobytes'):
                            b = img.tobytes()
                        else:
                            # PIL image
                            import io
                            buf = io.BytesIO()
                            img.save(buf, format='PNG')
                            b = buf.getvalue()
                        h = hashlib.md5(b).hexdigest()
                        hashes.add(h)
                    return hashes

                split_hashes = {}
                for split in splits:
                    print(f"Hashing {split} (first 1000)...")
                    split_hashes[split] = get_hashes(split, limit=1000)
                
                for s1, s2 in itertools.combinations(split_hashes.keys(), 2):
                    intersection = split_hashes[s1].intersection(split_hashes[s2])
                    if intersection:
                        print(f"WARNING: Found {len(intersection)} overlapping image hashes between {s1} and {s2} (in first 1000)!")
                        overlap_found = True
                    else:
                        print(f"No overlap found in hashes between {s1} and {s2}.")

    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    check_resisc()
