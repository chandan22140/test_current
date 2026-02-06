import math

def calculate_butterfly_params(b, k, d, verbose=True):
    """
    Calculate the number of trainable parameters for a single butterfly component.
    
    Following BOFT paper: A butterfly component B_tilde_b(d, k) has:
    - d/k blocks along the diagonal
    - Each block of size k×k uses k/(2b) orthogonal b×b sub-blocks
    - For b=2 (Givens rotations), each 2×2 block has 1 angle parameter
    
    Args:
        b (int): Block size (typically 2 for Givens rotations)
        k (int): Level size (power of 2: 2, 4, 8, ..., d)
        d (int): Full dimension of the rotation matrix
        
    Returns:
        int: Number of parameters for this butterfly component
        
    Example for d=128, k=64, b=2:
        - n_blocks = 128/64 = 2 (two BF(64) blocks on diagonal)
        - rotations_per_block = 64/2 = 32 (Givens pairs per BF block)
        - total = 2 * 32 = 64 parameters
    """
    if k < 2 or d < 2:
        return 0
    
    # Number of blocks = d/k
    n_blocks = d // k
    
    # For block_size=b, each BF(k) needs k/b rotation parameters
    # With b=2, this is k/2 Givens rotations per BF block
    rotations_per_block = k // b
    
    total_params = n_blocks * rotations_per_block
    
    if verbose:
        print(f"  ButterflyComponent(d={d}, k={k}, b={b}):")
        print(f"    n_blocks = d/k = {d}/{k} = {n_blocks}")
        print(f"    rotations_per_block = k/b = {k}/{b} = {rotations_per_block}")
        print(f"    params = {n_blocks} × {rotations_per_block} = {total_params}")
    
    return total_params


def calculate_full_butterfly_params(d, b=2, verbose=True):
    """
    Calculate total trainable parameters for a full butterfly rotation layer.
    
    A full ButterflyRotationLayer R(m, b) composes m = log2(d) butterfly components:
        R = B_tilde(d, d) @ B_tilde(d, d/2) @ ... @ B_tilde(d, 2)
    
    Each component has d/2 parameters (for b=2), so total = m * (d/2) = d * log2(d) / 2
    
    Args:
        d (int): Dimension (rank r). Should be power of 2 for exact calculation.
        b (int): Block size (default 2 for Givens rotations)
        
    Returns:
        int: Total parameters for one ButterflyRotationLayer
        
    Example for d=128, b=2:
        - m = log2(128) = 7 levels
        - Each level has d/2 = 64 parameters
        - Total = 7 × 64 = 448 parameters per R_U/R_V
    """
    if d < 2:
        return 0
    
    # Round up to nearest power of 2 if needed
    d_padded = 2 ** math.ceil(math.log2(d))
    
    # Number of levels m = log2(d)
    m = int(math.log2(d_padded))
    
    if verbose:
        print(f"\n=== Full Butterfly Params (d={d}, b={b}) ===")
        if d != d_padded:
            print(f"Note: d={d} padded to d_padded={d_padded}")
        print(f"m = log2({d_padded}) = {m} levels")
        print(f"\nBreakdown by level:")
    
    total = 0
    k = d_padded
    while k >= 2:
        params_k = calculate_butterfly_params(b, k, d_padded, verbose=verbose)
        total += params_k
        k = k // 2
    
    if verbose:
        print(f"\nTotal params per rotation matrix: {total}")
        # Verify with formula
        formula_total = d_padded * m // 2
        print(f"Formula check: d*log2(d)/2 = {d_padded}×{m}/2 = {formula_total}")
    
    return total


def calculate_pissa_params(num_layers, way, r, low_rank_r=None, use_butterfly=False, 
                           butterfly_sequential=False, verbose=True):
    """
    Calculate the number of trainable parameters for Rotational PiSSA.
    
    Args:
        num_layers (int): Number of linear layers applying the adapter.
        way (int or str): Method 'way0', 'way1', 'way2', 'way3'.
        r (int): Rank of the adapter.
        low_rank_r (int): Inner rank for Way 2/3 (default: 4 or similar).
        use_butterfly (bool): For Way 1, use butterfly factorization instead of Givens.
        butterfly_sequential (bool): For butterfly, train one component at a time.
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
        
    if way == "0":
        # Way 0: Two full rotation matrices R_U and R_V (r x r)
        # R_U: r*r, R_V: r*r
        params_per_layer = params_s + 2 * (r * r)
        desc = f"Way 0 (Full R): S({r}) + R_U({r}x{r}) + R_V({r}x{r})"
        
    elif way == "1":
        # Way 1 (SOARA-V2b / Sequential):
        # Formula: L * ceil(log2(r)) * (d + r)
        # where L = num_layers (total linear modules)
        #       m = ceil(log2(r))
        #       d = r (or r_padded) -> rotation params per phase
        #       r = singular values
        
        # Determine effective r (padded for butterfly)
        if use_butterfly:
             r_padded = 2 ** math.ceil(math.log2(r))
        else:
             r_padded = r 
             
        # Number of phases m
        if use_butterfly:
            m = math.ceil(math.log2(r_padded))
        else:
            m = r - 1
        
        # d in formula corresponds to rotation parameters per phase
        # (U and V each have d/2 angles -> total d)
        d_param = r_padded
        
        # params per phase = (d + r)
        params_per_phase = d_param + r
        
        if butterfly_sequential:
            # Sequential: only one phase active at a time
            # Formula: L * (d + r)
            params_per_layer = params_per_phase
            desc = f"Way 1 seq (formula): d({d_param}) + r({r})"
        else:
            # Full: all phases active
            # Formula: L * m * (d + r)
            params_per_layer = m * params_per_phase
            desc = f"Way 1 full (formula): m({m}) * (d({d_param}) + r({r}))"
        
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
        if way == "1" and use_butterfly:
            r_padded = 2 ** math.ceil(math.log2(r))
            m = int(math.log2(r_padded))
            print(f"Butterfly: {m} levels, r_padded={r_padded}")
            print(f"Butterfly Sequential: {butterfly_sequential}")
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
        parser = argparse.ArgumentParser(
            description="Calculate Rotational PiSSA trainable parameters",
            formatter_class=argparse.RawDescriptionHelpFormatter,
            epilog="""
Examples:
  # Standard Way 0 calculation
  python calculate_pissa_params.py --layers 72 --way 0 --rank 128

  # Way 1 with butterfly (all components)
  python calculate_pissa_params.py --layers 72 --way 1 --rank 128 --butterfly

  # Way 1 with butterfly sequential (one component active)
  python calculate_pissa_params.py --layers 72 --way 1 --rank 128 --butterfly --butterfly_sequential

  # Standalone butterfly calculation for custom b, k, d
  python calculate_pissa_params.py --butterfly_calc --d 128 --k 64 --b 2
  
  # Full butterfly layer params for rank d
  python calculate_pissa_params.py --butterfly_calc --d 128
""")
        parser.add_argument("--layers", type=int, default=72, help="Number of layers")
        parser.add_argument("--way", type=str, default="0", help="way0, way1, way2, way3")
        parser.add_argument("--rank", type=int, default=128, help="Rank r (dimension d)")
        parser.add_argument("--low_rank", type=int, default=4, help="Low rank for way 2/3")
        parser.add_argument("--butterfly", action="store_true", 
                           help="Use butterfly factorization for Way 1")
        parser.add_argument("--butterfly_sequential", action="store_true",
                           help="Train one butterfly component at a time (sequential)")
        
        # Standalone butterfly calculation
        parser.add_argument("--butterfly_calc", action="store_true",
                           help="Calculate butterfly params standalone (use with --d, --k, --b)")
        parser.add_argument("--d", type=int, default=128, help="Dimension d for butterfly")
        parser.add_argument("--k", type=int, default=None, 
                           help="Level k for single component (if omitted, calculates full layer)")
        parser.add_argument("--b", type=int, default=2, help="Block size b (default 2)")
        
        args = parser.parse_args()
        
        if args.butterfly_calc:
            # Standalone butterfly calculation mode
            if args.k is not None:
                # Single component
                print(f"\n=== Single Butterfly Component ===")
                params = calculate_butterfly_params(args.b, args.k, args.d, verbose=True)
                print(f"\nSingle component params: {params}")
            else:
                # Full butterfly layer
                params = calculate_full_butterfly_params(args.d, args.b, verbose=True)
                print(f"\n=== For Rotational PiSSA Way 1 Butterfly (2 rotation matrices) ===")
                print(f"R_U + R_V params: 2 × {params} = {2 * params}")
                print(f"With S (diagonal): {args.d} + {2 * params} = {args.d + 2 * params} params per layer")
        else:
            layers = args.layers
            way = args.way
            rank = args.rank
            low_rank = args.low_rank
            use_butterfly = args.butterfly
            butterfly_sequential = args.butterfly_sequential
            
            calculate_pissa_params(layers, way, rank, low_rank, use_butterfly, butterfly_sequential)
    else:
        print("Interactive Mode (pass arguments to skip)")
        try:
            l = input("Number of layers [72]: ").strip()
            layers = int(l) if l else 72
            
            w = input("Method Way (0/1/2/3) [0]: ").strip()
            way = w if w else "0"
            
            r = input("Rank r [128]: ").strip()
            rank = int(r) if r else 128
            
            use_butterfly = False
            butterfly_sequential = False
            if str(way) == "1":
                bf = input("Use butterfly factorization? (y/n) [n]: ").strip().lower()
                use_butterfly = bf == "y"
                if use_butterfly:
                    bfs = input("Butterfly sequential? (y/n) [n]: ").strip().lower()
                    butterfly_sequential = bfs == "y"
            
            if str(way) in ["2", "3"]:
                lr = input("Low Rank r' [4]: ").strip()
                low_rank = int(lr) if lr else 4
            else:
                low_rank = 4
        except ValueError:
            print("Invalid input, using defaults.")
            layers, way, rank, low_rank = 72, "0", 128, 4
            use_butterfly = False
            butterfly_sequential = False

        calculate_pissa_params(layers, way, rank, low_rank, use_butterfly, butterfly_sequential)

