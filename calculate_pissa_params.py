import math

def calculate_pissa_params(num_layers, way, r, low_rank_r=None, verbose=True):
    """
    Calculate the number of trainable parameters for Rotational PiSSA.
    
    Args:
        num_layers (int): Number of linear layers applying the adapter.
        way (int or str): Method 'way0', 'way1', 'way2', 'way3'.
        r (int): Rank of the adapter.
        low_rank_r (int): Inner rank for Way 2/3 (default: 4 or similar).
    """
    way = str(way).lower().replace("way", "")
    
    # Common parameters: S matrix (diagonal) is usually trainable (size r)
    # Assumes freeze_singular_values=False
    params_s = r
    
    params_per_layer = 0
    
    if way == "0":
        # Way 0: Two full rotation matrices R_U and R_V (r x r)
        # R_U: r*r, R_V: r*r
        params_per_layer = params_s + 2 * (r * r)
        desc = f"Way 0 (Full R): S({r}) + R_U({r}x{r}) + R_V({r}x{r})"
        
    elif way == "1":
        # Way 1: Sequential Givens Rotations (Batched SVD strategy)
        # Only ONE layer of disjoint pairs active at a time for U and V.
        # Max disjoint pairs in r dimensions = floor(r/2)
        # Active params = S + thetas_U + thetas_V
        n_pairs = r // 2
        params_per_layer = params_s + 2 * n_pairs
        desc = f"Way 1 (Sequential): S({r}) + Active_Givens_U({n_pairs}) + Active_Givens_V({n_pairs})"
        
    elif way == "2" or way == "3":
        # Way 2/3: Low-rank parameterization I + BC^T - CB^T
        # B_U, C_U, B_V, C_V all size (r x low_rank_r)
        if low_rank_r is None:
            low_rank_r = 4 # Default in config
            
        params_low_rank = 4 * (r * low_rank_r)
        params_per_layer = params_s + params_low_rank
        desc = f"Way {way} (Low-Rank): S({r}) + 4 * [r({r}) * lr({low_rank_r})]"
        
    else:
        return "Invalid Way"

    total_params = params_per_layer * num_layers
    
    if verbose:
        print(f"--- Configuration ---")
        print(f"Layers: {num_layers}")
        print(f"Method: Way {way}")
        print(f"Rank (r): {r}")
        if way in ["2", "3"]:
            print(f"Low Rank (r'): {low_rank_r}")
        print(f"--- Breakdown (per layer) ---")
        print(f"Expr: {desc}")
        print(f"Count: {params_per_layer:,}")
        print(f"--- Total ---")
        print(f"Total Trainable Params: {total_params:,}")
        print("---------------------")
        
    return total_params

if __name__ == "__main__":
    import argparse
    import sys

    # Check if any args were passed (other than script name)
    if len(sys.argv) > 1:
        parser = argparse.ArgumentParser()
        parser.add_argument("--layers", type=int, default=72, help="Number of layers")
        parser.add_argument("--way", type=str, default="0", help="way0, way1, way2, way3")
        parser.add_argument("--rank", type=int, default=128, help="Rank r")
        parser.add_argument("--low_rank", type=int, default=4, help="Low rank for way 2/3")
        args = parser.parse_args()
        
        layers = args.layers
        way = args.way
        rank = args.rank
        low_rank = args.low_rank
    else:
        print("Interactive Mode (pass arguments to skip)")
        try:
            l = input("Number of layers [72]: ").strip()
            layers = int(l) if l else 72
            
            w = input("Method Way (0/1/2/3) [0]: ").strip()
            way = w if w else "0"
            
            r = input("Rank r [128]: ").strip()
            rank = int(r) if r else 128
            
            if str(way) in ["2", "3"]:
                lr = input("Low Rank r' [4]: ").strip()
                low_rank = int(lr) if lr else 4
            else:
                low_rank = 4
        except ValueError:
            print("Invalid input, using defaults.")
            layers, way, rank, low_rank = 72, "0", 128, 4

    calculate_pissa_params(layers, way, rank, low_rank)
