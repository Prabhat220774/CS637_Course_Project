import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from scipy.optimize import minimize
from tqdm import tqdm

# --- THE FIX ---
# We must import the type hints we use in function signatures
from typing import List, Tuple, Callable
# --- END FIX ---


def calculate_ruc(invariant_series: pd.Series) -> pd.DataFrame:
    """
    Calculates the Safe Margins (Eq. 8, 9) and RUC (Eq. 10).
    
    This is the "Residual Under Curve" (RUC) which is the input
    to the final Hampel loss function.
    """
    
    # --- Hyperparameter from Paper (Sec III-E) ---
    # We choose an initial Epsilon (e) based on the paper's optimization.
    # A value around 2.5-3.0 seems optimal per their Fig 7b.
    EPSILON = 2.8
    
    # We group the invariant by the hour of the day
    hourly_groups = invariant_series.groupby(invariant_series.index.hour)
    
    # 1. Expected Value (Eq. 7)
    E_AD_t = hourly_groups.mean()
    
    # 2. Safe Range (Eq. 8, 9)
    # The paper uses MAD(AD(t)) - the MAD of the *entire* series.
    MAD_AD_t = (invariant_series - invariant_series.median()).abs().median()
    
    # Map the 24 hourly values back to the full time series
    E_AD_t_mapped = invariant_series.index.hour.map(E_AD_t)
    
    gamma_high = E_AD_t_mapped + EPSILON * MAD_AD_t
    gamma_low = E_AD_t_mapped - EPSILON * MAD_AD_t
    
    # 3. Formation of Stateless Residuals (RUC) (Eq. 10)
    ruc = pd.Series(0.0, index=invariant_series.index)
    
    # RUC(t) = AD(t) - Gamma_high(t)
    high_mask = invariant_series > gamma_high
    ruc[high_mask] = invariant_series[high_mask] - gamma_high[high_mask]
    
    # RUC(t) = AD(t) - Gamma_low(t)
    low_mask = invariant_series < gamma_low
    ruc[low_mask] = invariant_series[low_mask] - gamma_low[low_mask]
    
    return ruc.to_frame('ruc')

def find_htf_params(ruc_series: pd.Series) -> Tuple[float, float, float]:
    """
    Finds the 'a', 'b', 'c' params for Hampel Loss (Sec III-D).
    
    The paper does this by using K-Means (k=8) on the residuals
    and then combining the clusters to get 3 boundary points.
    """
    # We only care about the *magnitude* of residuals for the boundaries
    positive_ruc = ruc_series[ruc_series > 0].abs().values.reshape(-1, 1)
    
    if len(positive_ruc) < 8:
        # Not enough data, return sane defaults
        return 1.0, 2.0, 3.0
        
    # Run K-Means with k=8
    kmeans = KMeans(n_clusters=8, random_state=42, n_init=10)
    kmeans.fit(positive_ruc)
    
    # Get the 8 cluster centers and sort them
    centers = sorted(kmeans.cluster_centers_.flatten())
    
    # Per paper: "combine the clusters into 4 segments demarcated by 3 values"
    # We take the boundary points between clusters.
    # a = boundary between cluster 2 and 3
    # b = boundary between cluster 4 and 5
    # c = boundary between cluster 6 and 7
    # (Using median of centers as a robust boundary)
    
    a = (centers[1] + centers[2]) / 2
    b = (centers[3] + centers[4]) / 2
    c = (centers[5] + centers[6]) / 2

    # Ensure a < b < c and all > 0
    if not (0 < a < b < c):
        # Fallback to percentile-based estimation if clustering is weird
        percs = np.percentile(positive_ruc, [25, 50, 75])
        a, b, c = percs[0], percs[1], percs[2]
        if a == 0: a = 0.5 # avoid 0
        if b <= a: b = a + 0.5
        if c <= b: c = b + 0.5

    return a, b, c

def _hampel_cost_func(tau: float, ruc_values: np.ndarray, a: float, b: float, c: float) -> float:
    """
    The actual Hampel cost function (Eq. 11, simplified).
    This is what 'scipy.optimize.minimize' will try to minimize.
    
    Note: Paper's Algorithm 3 is a mess of cost *derivatives*.
    Eq. 11 is the *actual* loss function (p). We minimize the sum of p(s(t)).
    """
    # s(t) = r(t) - tau (where r(t) is RUC value)
    s = ruc_values - tau
    
    # --- Implementation of Eq. 11 ---
    # We only care about positive residuals for tau_max
    s = s[s > 0]
    if len(s) == 0:
        return 0.0

    cost = np.zeros_like(s)
    
    # |s(t)| <= a
    mask_a = (s <= a)
    cost[mask_a] = 0.5 * (s[mask_a]**2)
    
    # a < |s(t)| <= b
    mask_b = (s > a) & (s <= b)
    cost[mask_b] = (a * s[mask_b]) - (0.5 * a**2)
    
    # b < |s(t)| <= c
    mask_c = (s > b) & (s <= c)
    term_c = (c - s[mask_c]) / (c - b)
    cost[mask_c] = (a * b) - (0.5 * a**2) + (a * (c - b) / 2) * (1 - term_c**2)
    
    # |s(t)| > c
    mask_d = (s > c)
    cost[mask_d] = (a * b) - (0.5 * a**2) + (a * (c - b) / 2)

    return np.sum(cost)

def learn_robust_thresholds(ruc_df: pd.DataFrame) -> Tuple[float, float]:
    """
    Learns the tau_max and tau_min thresholds (Algorithm 3).
    """
    print("  Learning robust thresholds (Algorithm 3)...")
    
    # --- 1. Learn tau_max (for positive residuals) ---
    ruc_pos = ruc_df['ruc'][ruc_df['ruc'] > 0]
    
    if ruc_pos.empty:
        tau_max = 1.0 # Sane default
    else:
        # Find the a, b, c parameters
        a, b, c = find_htf_params(ruc_pos)
        
        # Initial guess for tau: 95th percentile (a robust start)
        initial_guess = np.percentile(ruc_pos, 95)
        
        # Run the optimization
        result = minimize(
            _hampel_cost_func,
            x0=[initial_guess],
            args=(ruc_pos.values, a, b, c),
            method='Nelder-Mead'
        )
        
        if result.success:
            tau_max = result.x[0]
        else:
            tau_max = initial_guess # Fallback
            
    # --- 2. Learn tau_min (for negative residuals) ---
    # We just mirror the process for the absolute value of negative residuals
    ruc_neg = ruc_df['ruc'][ruc_df['ruc'] < 0].abs()
    
    if ruc_neg.empty:
        tau_min_abs = 1.0 # Sane default
    else:
        a_n, b_n, c_n = find_htf_params(ruc_neg)
        initial_guess_n = np.percentile(ruc_neg, 95)
        
        result_n = minimize(
            _hampel_cost_func,
            x0=[initial_guess_n],
            args=(ruc_neg.values, a_n, b_n, c_n),
            method='Nelder-Mead'
        )
        
        if result_n.success:
            tau_min_abs = result_n.x[0]
        else:
            tau_min_abs = initial_guess_n
            
    # Remember, tau_min is the *negative* boundary
    tau_min = -tau_min_abs
    
    print(f"  Thresholds learned: (tau_min: {tau_min:.4f}, tau_max: {tau_max:.4f})")
    return tau_max, tau_min


