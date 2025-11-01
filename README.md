# CS637 Course Project: SWM Anomaly Detection

This project is a full, production-grade Python implementation of the 3-stage anomaly detection pipeline described in the paper:
**"Unsafe Events Detection in Smart Water Meter Infrastructure Via Noise-Resilient Learning"** (ICCPS 2024).

This implementation is designed to run on a large-scale (1.1GB, 16M+ rows) real-world dataset and includes the paper's novel **Residency Similarity Recognition (RSR)** clustering (Algorithm 1) and the **Hampel Loss (HTF)** robust thresholding (Algorithm 3).

## 1. Data Setup (CRITICAL)

This repository **does not** contain the 1.1GB data file. It must be downloaded separately.

1.  **Download the data (`swm_trialA_1K.csv`) from here:**
    `https://github.com/DAIAD/data/blob/master/swm_trialA_1K.zip` 

2.  **Place the file** in the `data/` directory.

The final structure should be: `cs637_swm_anamoly/data/swm_trialA_1k.csv`

## 2. How to Run

This pipeline is split into two main commands: running the model and evaluating the results.

### Step 1: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 2: Run the Model (This will take time)

This command runs the entire data processing, clustering, and anomaly detection pipeline.
```bash
python3 main.py
```
### NOTE: The first time you run this, it will be very slow.

1. Data Loading: It will process the 1.1GB file (5-10 mins).

2. Clustering: It will build a 1099x1099 Mutual Information matrix using all available CPU cores. This can take several hours.

The script is smart. It caches all this hard work in the data/ folder. Every subsequent run will be instantaneous (it will load from the cache).

## 3. The Debugging Journey: From CSV to Pipeline

This project was a real-world data science challenge. The initial 1.1 GB file was **hostile** and required several layers of debugging to load successfully.  
The `src/data_loader.py` script is the result of solving all of these issues:

1. **The Separator**:  
   The file was a **Semicolon-Separated Value (SSV)** file, not a CSV.  
   The fix was: `sep=';'`.

2. **The Encoding**:  
   The file contained an invisible **Byte Order Mark (BOM)**, which hid the first column’s name.  
   The fix was: `encoding='utf-8-sig'`.

3. **The Column Names**:  
   The headers contained spaces (e.g., `user key`), not underscores.

4. **The Date Format**:  
   The `datetime` column used the **Day-First format (`DD/MM/YYYY`)**, not the default (`MM/DD/YYYY`).  
   The fix was: `dayfirst=True`.

5. **The Memory Limit**:  
   The initial pivot logic tried to create a **16-billion-cell matrix**, causing a `MemoryError`.  
   The fix was to re-architect the pivot to a memory-efficient  
   `groupby().resample().unstack()` pipeline.


## 4. Project Structure
```
/
|-- main.py                 <-- SCRIPT 1: Runs the full model
|-- evaluate.py             <-- SCRIPT 2: Runs the evaluation
|-- requirements.txt
|-- README.md               <-- This file
|-- .gitignore              <-- Tells Git to ignore the 1.1GB data file
|
|-- data/
|   |-- swm_trialA_1k.csv     <-- (You must add this file here (it's small 'k' not captital 'K'))
|
|-- src/
|   |-- __init__.py
|   |-- data_loader.py        (Handles the hostile CSV/SSV)
|   |-- clustering.py         (Implements Algorithm 1: RSR)
|   |-- feature_engineering.py (Implements Equation 6: The AD(t) Invariant)
|   |-- anomaly_detector.py   (Implements Algorithm 3: HTF)
|   |-- check_columns.py      (The diagnostic tool we used)
```
