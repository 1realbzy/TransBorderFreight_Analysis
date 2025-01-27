# scripts/process_january_data.py
import os
import pandas as pd

# Define paths
DATA_DIR = r"C:\Users\hbempong\TransBorderFreight_Analysis\data\2020\Jan 2020"
OUTPUT_DIR = r"C:\Users\hbempong\TransBorderFreight_Analysis\output"

def process_csv(file_path):
    """Read, summarize, and save the processed data."""
    df = pd.read_csv(file_path)
    summary = df.describe()  # Summarize numeric data
    
    output_file = os.path.join(OUTPUT_DIR, f"summary_{os.path.basename(file_path)}")
    summary.to_csv(output_file, index=False)
    print(f"Processed and saved summary to {output_file}")

def process_january_data():
    """Process all CSV files in January 2020 data directory."""
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
    
    for file_name in os.listdir(DATA_DIR):
        if file_name.endswith(".csv"):
            file_path = os.path.join(DATA_DIR, file_name)
            print(f"Processing {file_name}...")
            process_csv(file_path)

if __name__ == "__main__":
    process_january_data()
