import pandas as pd
from pathlib import Path
import sys

# ---
# CONFIGURATION
# ---
DATA_FILE_PATH = Path("data/swm_trialA_1k.csv")
# ---

def diagnose_header():
    """
    Loads only the header of the CSV, cleans it, and prints the
    true column names to solve the configuration mystery.
    
    This version now explicitly uses a semicolon (';') separator.
    """
    
    print("***************************************************")
    print("* CSV Header Diagnostic Tool (Semicolon Mode)   *")
    print("***************************************************")
    
    if not DATA_FILE_PATH.exists():
        print(f"\nFATAL: Data file not found at '{DATA_FILE_PATH}'")
        print("Please make sure your CSV is in the 'data/' folder.")
        sys.exit(1)
        
    print(f"\nReading header from: {DATA_FILE_PATH}")

    try:
        # Load *only the first row* (nrows=0) to get the header.
        # We add sep=';' to correctly split the columns.
        df = pd.read_csv(
            DATA_FILE_PATH, 
            nrows=0, 
            encoding="utf-8-sig",
            sep=';' # <-- THE FIX
        )
        
        # Get the raw column names
        raw_columns = df.columns
        
        # Clean the column names (strip leading/trailing spaces)
        cleaned_columns = [str(col).strip() for col in raw_columns]
        
        print(f"\nFound {len(raw_columns)} columns (split by ';').")
        print("Here are the CLEANED names for your config:")
        
        print("\n--- COPY-PASTE THESE ---")
        for i, (raw, clean) in enumerate(zip(raw_columns, cleaned_columns), 1):
            print(f"Column {i}: \"{clean}\"")
            if raw != clean:
                print(f"       (Original was: \"{raw}\")")
        print("--------------------------\n")

        print("Update the CONFIGURATION block at the top of 'src/data_loader.py' ")
        print("to *exactly* match these cleaned names.")
        
    except Exception as e:
        print(f"\nFATAL: An error occurred while reading the header:")
        print(f"{e}")

if __name__ == "__main__":
    diagnose_header()


