import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm

# Import the data loader from our 'src' package
from src.data_loader import load_and_pivot_data

# ---
# CONFIGURATION - Must match main.py and the paper!
# ---

# 1. Data/Split Configuration (must be identical to main.py)
DATA_FILE_PATH = Path("data/swm_trialA_1k.csv")
TRAIN_END_DATE = "2016-06-01" # All data after this is the "Test Set"

# 2. Ground Truth Configuration (from Paper, Sec IV-F-4)
THETA_L = -1.0    # Low threshold (e.g., backflow)
THETA_H = 740.7   # High threshold (e.g., leaks)

# 3. Our Model's Results
MODEL_REPORT_FILE = Path("data/anomalies_report.csv")

def generate_ground_truth(data_test: pd.DataFrame) -> pd.Series:
    """
    Generates the Ground Truth by applying the paper's rules (Eq. 14).
    
    Args:
        data_test: The (T_test x N) matrix of all user readings.
        
    Returns:
        A pandas Series with the index as datetime, where a value of
        True means that timestamp is a ground truth anomaly.
    """
    print("Generating Ground Truth from test set (as per Eq. 14)...")
    
    # Check for high anomalies (any reading > 740.7)
    # .any(axis=1) checks row-by-row if any user at that time was an anomaly
    high_anomalies = (data_test > THETA_H).any(axis=1)
    
    # Check for low anomalies (any reading < -1.0)
    low_anomalies = (data_test < THETA_L).any(axis=1)
    
    # Combine them. A timestamp is an anomaly if it's high OR low.
    ground_truth = high_anomalies | low_anomalies
    
    # Filter to only the anomalous timestamps
    ground_truth_anomalies = ground_truth[ground_truth]
    
    print(f"Ground Truth generated. Found {len(ground_truth_anomalies)} anomalous time steps.")
    
    # Save for inspection (optional)
    ground_truth_anomalies.to_csv("data/ground_truth_anomalies.csv")
    
    return ground_truth_anomalies

def main():
    """
    Main evaluation script.
    Compares our model's output to the paper's ground truth rules.
    """
    print("===== SWM Anomaly Evaluation Script =====")
    
    # --- 1. Load Data Matrix ---
    try:
        data_matrix = load_and_pivot_data(DATA_FILE_PATH)
        if data_matrix.empty: return
    except FileNotFoundError:
        print(f"ERROR: Data file not found. Put '{DATA_FILE_PATH.name}' in '{DATA_FILE_PATH.parent}'")
        return
        
    # --- 2. Get Test Set ---
    data_test = data_matrix.loc[TRAIN_END_DATE:]
    if data_test.empty:
        print(f"ERROR: No test data found after {TRAIN_END_DATE}.")
        return
        
    # --- 3. Generate Ground Truth ---
    # This is the "answer key"
    gt_anomalies = generate_ground_truth(data_test)
    total_gt_anomalies = len(gt_anomalies)
    
    if total_gt_anomalies == 0:
        print("ERROR: No ground truth anomalies found. Check thresholds.")
        return

    # --- 4. Load Our Model's Report ---
    try:
        model_report = pd.read_csv(MODEL_REPORT_FILE, parse_dates=['datetime'], index_col='datetime')
    except FileNotFoundError:
        print(f"ERROR: Model report '{MODEL_REPORT_FILE}' not found.")
        print("Please run 'python main.py' first to generate the report.")
        return
        
    # Get unique anomalous timestamps from our model
    model_anomalies = model_report.index.unique()
    total_model_detections = len(model_anomalies)
    
    print(f"Model report loaded. Found {total_model_detections} anomalous time steps.")

    # --- 5. Compare and Score ---
    print("\n===== Final Evaluation =====")
    
    # True Positives (TP): Our model found an anomaly, and it WAS in the ground truth
    # We check if our model's timestamps are present in the ground truth's index
    tp_mask = model_anomalies.isin(gt_anomalies.index)
    true_positives = len(model_anomalies[tp_mask])
    
    # False Positives (FP): Our model found an anomaly, but it was NOT in the ground truth
    fp_mask = ~model_anomalies.isin(gt_anomalies.index)
    false_positives = len(model_anomalies[fp_mask])
    
    # False Negatives (FN): We MISSED an anomaly that was in the ground truth
    fn_mask = ~gt_anomalies.index.isin(model_anomalies)
    false_negatives = len(gt_anomalies.index[fn_mask])

    # --- 6. Calculate Paper's Metrics (Fig. 8) ---
    
    # Detection Rate (True Positive Rate / Recall)
    # = (Anomalies we found) / (All real anomalies)
    detect_rate = true_positives / total_gt_anomalies
    
    # Missed Detection Rate (False Negative Rate)
    # = (Anomalies we missed) / (All real anomalies)
    md_rate = false_negatives / total_gt_anomalies
    
    # False Alarm Rate (False Positive Rate)
    # = (Anomalies we *falsely* claimed) / (All *benign* time steps)
    total_test_steps = len(data_test)
    total_benign_steps = total_test_steps - total_gt_anomalies
    fa_rate = false_positives / total_benign_steps

    print(f"Total Time Steps in Test Set: {total_test_steps}")
    print(f"Total Benign Steps:           {total_benign_steps}")
    print(f"Total Ground Truth Anomalies: {total_gt_anomalies} (from Eq. 14)\n")
    
    print(f"True Positives (TP):   {true_positives}")
    print(f"False Positives (FP):  {false_positives}")
    print(f"False Negatives (FN):  {false_negatives}\n")
    
    print("--- Your Project Score (Compare to Fig. 8) ---")
    print(f"Detection Rate (Recall): {detect_rate:.2%}")
    print(f"Missed Detection Rate:   {md_rate:.2%}")
    print(f"False Alarm Rate (FPR):  {fa_rate:.2%}")

if __name__ == "__main__":
    main()

