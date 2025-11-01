import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm
import warnings

# ---
# Import our custom modules
# ---
from src.data_loader import load_and_pivot_data, save_data, data_exists
from src.clustering import run_rsr_clustering, clustering_exists, save_clustering
from src.feature_engineering import calculate_invariant
# --- THE FIX (Line 9) ---
# We are now importing 'calculate_ruc' and 'learn_robust_thresholds'
from src.anomaly_detector import (
    calculate_ruc,
    learn_robust_thresholds,
)
# --- END FIX ---


# ---
# Configuration
# ---
DATA_FILE_PATH = Path("data/swm_trialA_1k.csv")
DATA_MATRIX_CACHE = Path("data/data_matrix.pkl")
CLUSTER_CACHE = Path("data/cluster_assignments.json")
REPORT_FILE = Path("data/anomalies_report.csv")

# Per paper, data before June 2016 is for training
TRAIN_END_DATE = "2016-06-01"


def main():
    """
    Main driver script to run the full pipeline.
    """
    # Suppress warnings (e.g., from K-Means n_init)
    warnings.filterwarnings("ignore", category=FutureWarning)
    warnings.filterwarnings("ignore", category=UserWarning)

    print("======================================================")
    print(" SWM Anomaly Detection - Full Pipeline (ICCPS 2024)")
    print("======================================================")

    # ---
    # STAGE 1: DATA LOADING
    # ---
    if data_exists(DATA_MATRIX_CACHE):
        print(f"Found cached data matrix. Loading '{DATA_MATRIX_CACHE}'...")
        data_matrix = pd.read_pickle(DATA_MATRIX_CACHE)
    else:
        print(f"No cache found. Loading and pivoting 1.1GB CSV...")
        try:
            data_matrix = load_and_pivot_data(DATA_FILE_PATH)
            save_data(data_matrix, DATA_MATRIX_CACHE)
        except FileNotFoundError:
            print(f"FATAL: Data file not found at '{DATA_FILE_PATH}'")
            print("Please put your 1.1GB CSV in the 'data/' folder.")
            return
        except Exception as e:
            print(f"FATAL: Could not load data. Error: {e}")
            return

    print(f"Data matrix loaded. Shape: {data_matrix.shape} (Time x Users)")

    # Split into Train/Test
    data_train = data_matrix.loc[:TRAIN_END_DATE]
    # data_test = data_matrix.loc[TRAIN_END_DATE:] # Not needed for this script

    if data_train.empty:
        print(f"FATAL: No training data found before {TRAIN_END_DATE}.")
        return

    # ---
    # STAGE 2: CLUSTERING (Algorithm 1: RSR)
    # ---
    if clustering_exists(CLUSTER_CACHE):
        print(f"Found cached cluster assignments. Loading '{CLUSTER_CACHE}'...")
        clusters = pd.read_json(CLUSTER_CACHE, typ="series").to_dict()
        # Convert string keys back to int
        clusters = {int(k): v for k, v in clusters.items()}
    else:
        print("No cluster cache found. Running RSR Clustering (Algorithm 1)...")
        print("This will take a long time and use all your CPU cores.")
        clusters = run_rsr_clustering(data_train)
        save_clustering(clusters, CLUSTER_CACHE)

    print(f"Clustering complete. Found {len(clusters)} clusters.")

    # ---
    # STAGE 3: ANALYSIS (Invariant + Anomaly Detection)
    # ---
    print("\nStarting main analysis loop (Stage 3)...")
    all_anomalies_list = []
    
    # We need to learn thresholds from ALL training RUCs, not per-cluster
    all_ruc_dfs = {}
    
    # --- Sub-Stage 3a: Calculate Invariant and RUC for all clusters
    
    for cluster_id, user_keys in tqdm(clusters.items(), desc="Calculating Invariants"):
        if len(user_keys) < 2:
            continue # Skip clusters with one user
            
        cluster_train_data = data_train[user_keys].dropna()
        if cluster_train_data.empty:
            continue

        # 1. Calculate Invariant (Eq. 6)
        invariant_df = calculate_invariant(cluster_train_data)
        
        # 2. Calculate Safe Margins (RUC) (Eq. 8, 9, 10)
        invariant_df = invariant_df.dropna()
        
        # --- THE FIX (Line 105) ---
        # Calling the new, correct function name
        ruc_df = calculate_ruc(invariant_df["invariant"])
        # --- END FIX ---
        
        all_ruc_dfs[cluster_id] = ruc_df

    # --- Sub-Stage 3b: Learn Robust Thresholds (Alg. 3)
    # We learn the thresholds from the *entire* training set's RUCs
    
    print("\nLearning robust thresholds (Algorithm 3) from all training data...")
    full_train_ruc = pd.concat(all_ruc_dfs.values())
    tau_max, tau_min = learn_robust_thresholds(full_train_ruc)
    
    # --- Sub-Stage 3c: Find Anomalies in Test Set
    print("Scanning test set for anomalies...")
    
    # We load the FULL matrix again to process the test set
    # This is more efficient than loading it twice
    data_test = data_matrix.loc[TRAIN_END_DATE:]

    for cluster_id, user_keys in tqdm(clusters.items(), desc="Finding Anomalies"):
        if cluster_id not in all_ruc_dfs: # Skip clusters we didn't process
            continue
            
        cluster_test_data = data_test[user_keys].dropna()
        if cluster_test_data.empty:
            continue

        # 1. Calculate Invariant on TEST data
        invariant_test_df = calculate_invariant(cluster_test_data)
        
        # 2. Calculate RUC on TEST data
        ruc_test_df = calculate_ruc(invariant_test_df["invariant"])
        
        # 3. Apply thresholds (Eq. 12)
        anomalies = ruc_test_df[
            (ruc_test_df['ruc'] > tau_max) | (ruc_test_df['ruc'] < tau_min)
        ]
        
        if not anomalies.empty:
            anomalies['cluster_id'] = cluster_id
            anomalies['tau_max'] = tau_max
            anomalies['tau_min'] = tau_min
            all_anomalies_list.append(anomalies)

    # ---
    # STAGE 4: REPORTING
    # ---
    print("\n======================================================")
    print(" Pipeline Finished. Final Report:")
    print("======================================================")
    
    if not all_anomalies_list:
        print("No anomalies found in the test set.")
        return

    final_report = pd.concat(all_anomalies_list)
    final_report = final_report.sort_index()
    
    print(f"Found {len(final_report)} anomalous time steps in the test set.")
    
    # Save the report
    final_report.to_csv(REPORT_FILE)
    print(f"Full report saved to '{REPORT_FILE}'")
    
    print("\nSample of anomalies (first 20):")
    print(final_report.head(20))


if __name__ == "__main__":
    main()

