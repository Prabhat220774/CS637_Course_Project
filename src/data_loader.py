import pandas as pd
from pathlib import Path
from tqdm import tqdm
import warnings

# ---
# CONFIGURATION:
# ---
USER_COL = "user key"
TIME_COL = "datetime"
VALUE_COL = "diff"
# ---

def load_and_pivot_data(filepath: Path) -> pd.DataFrame:
    """
    Loads the 1.1GB SSV (Semicolon Separated) file and pivots it
    using a memory-efficient groupby-resample-unstack strategy.
    """
    print(f"  Starting 1.1GB SSV load from '{filepath.name}'...")
    
    warnings.simplefilter(action='ignore', category=pd.errors.DtypeWarning)
    
    chunk_list = []
    use_cols = [USER_COL, TIME_COL, VALUE_COL]
    dtypes = {
        USER_COL: "string", 
        TIME_COL: "object",
        VALUE_COL: "float32"
    }
    
    try:
        iterator = pd.read_csv(
            filepath,
            usecols=use_cols,
            dtype=dtypes,
            chunksize=1_000_000, 
            encoding="utf-8-sig",
            sep=';'
        )
        
        print("  Reading chunks (this may take a minute)...")
        for chunk in tqdm(iterator, desc="    Reading chunks", unit="M rows"):
            chunk[TIME_COL] = pd.to_datetime(chunk[TIME_COL], dayfirst=True)
            chunk_list.append(chunk)

    except Exception as e:
        print(f"\n--- FATAL: Error during CSV read ---")
        print(f"Error: {e}")
        print("Please check data file and configuration.")
        print("--------------------------------------\n")
        raise e

    print("\n  All chunks read. Concatenating...")
    df = pd.concat(chunk_list, ignore_index=True)
    chunk_list = None # Free memory
    
    print(f"  Raw data loaded. Shape: {df.shape}")
    print("  Pivoting data using memory-efficient groupby-resample-unstack...")

    # ---
    # THIS IS THE NEW, MEMORY-EFFICIENT LOGIC
    # ---
    
    # 1. Set the datetime index (on the 16.8M row dataframe)
    df = df.set_index(TIME_COL)
    
    # 2. Group by user, then resample each group to 1H.
    # This is the magic. It creates a small, resampled "long" series.
    print("  Step 1/2: Grouping by user and resampling to 1-hour grid...")
    
    # We apply .mean() to average out any sub-hour readings
    long_resampled_series = df.groupby(USER_COL)[VALUE_COL].resample('1H').mean()
    df = None # Free memory
    
    # 3. Unstack the *small, resampled* series.
    # This directly creates the final (T x N) matrix.
    print("  Step 2/2: Unstacking to final (Time x User) matrix...")
    data_matrix = long_resampled_series.unstack(level=0) # Unstack the USER_COL
    
    print("  Pivot complete. Cleaning final matrix...")
    
    # Clean up the data as per the paper's logic
    data_matrix = data_matrix.clip(lower=0)
    data_matrix = data_matrix.ffill().bfill()
    data_matrix = data_matrix.astype("float32")
    
    print(f"  Data matrix is clean and ready. Final Shape: {data_matrix.shape}")
    return data_matrix

# ---
# HELPER FUNCTIONS (No changes)
# ---

def save_data(data_matrix: pd.DataFrame, cache_path: Path):
    """
    Saves the processed (T x N) DataFrame to a pickle file.
    """
    print(f"  Caching data matrix to '{cache_path}'...")
    try:
        data_matrix.to_pickle(cache_path)
        print("  Caching complete.")
    except Exception as e:
        print(f"  Error caching data: {e}")

def data_exists(cache_path: Path) -> bool:
    """
    Checks if the data cache file already exists.
    """
    return cache_path.exists()


