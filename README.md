# SWM Anomaly Detection Pipeline (CS637 Project)

This project implements the full, 3-stage data science pipeline described in the paper: "Unsafe Events Detection in Smart Water Meter Infrastructure Via Noise-Resilient Learning" (ICCPS 2024).

It is designed to be a robust, production-style implementation, capable of handling large datasets (like the 1.1GB `swm_trialA_1k.csv`).

## Pipeline Stages

1.  **Data Loading (`src.data_loader`)**:
    * Loads the 1.1GB CSV.
    * Pivots the "long" data (1-row-per-reading) into a "wide" data matrix `(T x N)`, where `T` is the number of time steps (hours) and `N` is the number of users (SWMs).
    * This is a memory-intensive but necessary step to enable matrix-based calculations.

2.  **Clustering (`src.clustering`)**:
    * Implements the paper's novel **Residency Similarity Recognition (RSR)** algorithm (Algorithm 1).
    * This is the most computationally expensive part of the pipeline.
    * It first computes an `N x N` Mutual Information (MI) matrix in parallel.
    * It then uses `networkx` to build a graph and implements the paper's partitioning logic (Eq. 2, 3) to find the clusters.

3.  **Analysis (`src.feature_engineering` & `src.anomaly_detector`)**:
    * **Invariant Calculation**: Implements the `AD(t)` invariant (Eq. 6) using Generalized Means (Eq. 4).
    * **Robust Thresholding**: Implements the full **Hampel Three-part Loss Function (HTF)** (Algorithm 3).
        * This includes finding the `a, b, c` parameters by clustering the `RUC(t)` residuals (Eq. 10).
        * It uses `scipy.optimize.minimize` to find the optimal `tau` that minimizes the total Hampel cost function, exactly as described in the paper.

## How to Run

1.  **Place Data**: Put your `swm_trialA_1k.csv` file into the `data/` directory.

2.  **Install Dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

3.  **Run the Pipeline**:
    ```bash
    python main.py
    ```

**WARNING**: This is a heavy-duty data processing pipeline. The **Clustering** step (calculating the MI matrix) will take a significant amount of time (potentially 30 mins to hours, depending on your CPU) and will use all available CPU cores.

The output will be saved to `data/anomalies_report.csv` and `data/cluster_assignments.json`.
