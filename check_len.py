import pickle
import os

cache_filename = "/home/chandan/test_current/data_cache/load_meta_math_full_b14a7b8059d9c055954c92674ce60032.pkl"
with open(cache_filename, "rb") as f:
    data = pickle.load(f)
    train_set = data[0]
    print(f"Train set length: {len(train_set)}")
