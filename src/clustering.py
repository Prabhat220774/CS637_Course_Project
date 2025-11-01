import pandas as pd
import numpy as np
from sklearn.metrics import mutual_info_score
from joblib import Parallel, delayed
import networkx as nx
from tqdm import tqdm
from pathlib import Path
import json
import warnings

def _calculate_mi_chunk(user_keys: list, data: pd.DataFrame, start_idx: int) -> list:
    """
    Calculates a chunk of the Mutual Information (MI) matrix.
    This is the target for parallel processing.
    
    Args:
        user_keys: List of all user_key column names.
        data: The (T x N) training data matrix.
        start_idx: The starting row index for this chunk.
    
    Returns:
        A list of tuples (user_i, user_j, mi_score) for this chunk.
    """
    chunk_edges = []
    n_users = len(user_keys)
    
    # We only compute the upper triangle of the matrix (i > j)
    for i in range(start_idx, n_users):
        for j in range(i + 1, n_users):
            user_i_key = user_keys[i]
            user_j_key = user_keys[j]
            
            # Discretize data to compute MI (MI works on discrete distributions)
            # We'll bin into 10 bins, a common practice.
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", category=RuntimeWarning)
                    
                    bins_i = pd.qcut(data[user_i_key], 10, labels=False, duplicates='drop')
                    bins_j = pd.qcut(data[user_j_key], 10, labels=False, duplicates='drop')
                
                    mi = mutual_info_score(bins_i, bins_j)
                    
                    if mi > 0: # Only store positive correlations
                        chunk_edges.append((user_i_key, user_j_key, mi))
            except Exception:
                # This can fail if a user has zero variance (all same values)
                continue
                
    return chunk_edges

def run_rsr_clustering(data: pd.DataFrame) -> dict:
    """
    Implements the full Residency Similarity Recognition (RSR) alg (Alg 1).
    
    This is the most computationally expensive part of the pipeline.
    """
    
    # --- Hyperparameters from Paper (Sec II-C) ---
    W_CUT = 0.061 # Minimum Mutual Information to be considered an edge
    N_MAX = 80    # Max users per cluster
    # ---
    
    user_keys = list(data.columns)
    n_users = len(user_keys)
    
    print(f"  Building Mutual Information (MI) matrix for {n_users} users...")
    print(f"  This will use all available CPU cores...")
    
    # --- Parallel MI Matrix Calculation ---
    # We farm out rows of the matrix to different cores
    # n_jobs=-1 means use ALL available cores
    results = Parallel(n_jobs=-1)(
        delayed(_calculate_mi_chunk)(user_keys, data, i)
        for i in tqdm(range(n_users), desc="  MI Matrix (Parallel)")
    )
    
    # Flatten the list of lists into a single list of edges
    all_edges = [edge for chunk in results for edge in chunk]
    
    print(f"\n  MI calculation complete. Found {len(all_edges)} potential edges.")
    print("  Building graph and applying RSR (Algorithm 1)...")
    
    # --- Build the Graph (Section II-B-3) ---
    G = nx.Graph()
    G.add_nodes_from(user_keys)
    
    # Add only edges that meet the W_CUT threshold
    valid_edges = [
        (u, v, {"weight": w}) for u, v, w in all_edges if w > W_CUT
    ]
    G.add_edges_from(valid_edges)
    
    # --- Implement Algorithm 1 ---
    clusters = {}
    cluster_id = 0
    
    # Keep track of nodes we've already clustered
    clustered_nodes = set()
    
    # We sort edges by weight to start with the strongest connections
    # This is a robust modification of the paper's "random edge" start
    sorted_edges = sorted(G.edges(data=True), key=lambda x: x[2]['weight'], reverse=True)
    
    pbar = tqdm(total=n_users, desc="  Running RSR Clustering")
    
    for u, v, data in sorted_edges:
        if u in clustered_nodes or v in clustered_nodes:
            continue
            
        # 1. Start a new cluster (Lines 2-6)
        c = {u, v}
        vol_c = data['weight']
        
        # Calculate sum_cut (Eq. 3)
        sum_cut_c = sum(G[n][neighbor]['weight'] for n in c for neighbor in G[n] if neighbor not in c)
        
        clustered_nodes.update(c)
        pbar.update(2)
        
        # 2. Grow the cluster (Lines 7-15)
        while vol_c <= sum_cut_c and len(c) < N_MAX:
            # Find all neighbors (Line 8)
            neighbors = set(n for node in c for n in G[node] if n not in c)
            if not neighbors:
                break # No more neighbors to add
            
            # Find best neighbor to add (Line 12)
            # "selects the edge with the highest w_e from Neighbors(c)"
            best_neighbor = None
            max_edge_weight = -1
            
            for n in neighbors:
                # Weight of edges connecting this neighbor *to the cluster*
                edge_weight_to_c = sum(G[n][node]['weight'] for node in c if node in G[n])
                if edge_weight_to_c > max_edge_weight:
                    max_edge_weight = edge_weight_to_c
                    best_neighbor = n
            
            if best_neighbor is None or max_edge_weight == 0:
                break # No valid neighbors left to add
            
            # Add the new node to the cluster (Line 13-15)
            c.add(best_neighbor)
            clustered_nodes.add(best_neighbor)
            pbar.update(1)
            
            # Recalculate vol_c and sum_cut_c for the *new* cluster c
            vol_c = sum(G[u][v]['weight'] for u in c for v in c if v > u and v in G[u])
            sum_cut_c = sum(G[n][neighbor]['weight'] for n in c for neighbor in G[n] if neighbor not in c)

        # 3. Finalize cluster (Lines 16-17)
        clusters[cluster_id] = list(c)
        cluster_id += 1
        
    pbar.close()
    
    print(f"\n  RSR clustering finished. Found {len(clusters)} clusters.")
    return clusters

# ---
# NEW HELPER FUNCTIONS
# ---

def clustering_exists(cache_path: Path) -> bool:
    """
    Checks if the cluster assignment JSON cache file already exists.
    """
    return cache_path.exists()

def save_clustering(clusters: dict, cache_path: Path):
    """
    Saves the cluster assignments to a JSON file.
    """
    print(f"  Caching cluster assignments to '{cache_path}'...")
    try:
        # Convert int keys to str for JSON compatibility
        clusters_str_keys = {str(k): v for k, v in clusters.items()}
        with open(cache_path, 'w') as f:
            json.dump(clusters_str_keys, f, indent=4)
        print("  Caching complete.")
    except Exception as e:
        print(f"  Error caching clusters: {e}")


