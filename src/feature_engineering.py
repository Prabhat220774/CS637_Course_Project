import numpy as np
import pandas as pd
from scipy.stats import gmean

def generalized_mean(data: np.ndarray, p: float) -> float:
    """
    Implements the Generalized Mean (GM) from Equation (4).
    
    Handles special cases and ensures numerical stability.
    """
    # Paper's theory (VI-E) relies on positive consumption.
    # Clip at a small epsilon to avoid log(0) or /0 errors.
    epsilon = 1e-9
    data = np.clip(data, epsilon, np.inf)
    
    if p == 0:
        return gmean(data)
    if p == 1:
        return np.mean(data)
    
    return np.power(np.mean(np.power(data, p)), 1/p)

def calculate_invariant(cluster_data: pd.DataFrame) -> pd.DataFrame:
    """
    Implements the paper's core invariant 'AD(t)' from Equation (6).
    
    AD(t) = |GM(p_neg) - GM(p_pos)|
    
    Args:
        cluster_data: A (T x k) DataFrame for a single cluster.
        
    Returns:
        A (T x 1) DataFrame (a Series) of the invariant.
    """
    # --- Hyperparameter Selection ---
    # From paper's theory (Sec VI-E):
    # p_pos (Schur-convex, p >= 1): Senses large values (leaks)
    # p_neg (Schur-concave, p < 0): Senses small values (backflow)
    # We choose defensible values:
    P_POS = 2.0
    P_NEG = -2.0
    
    # --- Implementation ---
    # .apply(..., axis=1) iterates over each row (time step).
    
    # 1. Calculate GM with positive order (p+)
    gm_pos = cluster_data.apply(
        lambda row: generalized_mean(row.values, P_POS),
        axis=1
    )
    
    # 2. Calculate GM with negative order (p-)
    gm_neg = cluster_data.apply(
        lambda row: generalized_mean(row.values, P_NEG),
        axis=1
    )
    
    # 3. Calculate the invariant (Eq. 6)
    invariant_series = (gm_neg - gm_pos).abs()
    invariant_series.name = 'invariant'
    
    return invariant_series.to_frame()


