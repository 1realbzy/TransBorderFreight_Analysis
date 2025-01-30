# scripts/clean_summary.py

import os
import pandas as pd

# Directory containing the summary files
OUTPUT_DIR = r"C:\Users\hbempong\TransBorderFreight_Analysis\output"
CLEANED_OUTPUT_DIR = os.path.join(OUTPUT_DIR, "cleaned")

# Ensure cleaned output directory exists
if not os.path.exists(CLEANED_OUTPUT_DIR):
    os.makedirs(CLEANED_OUTPUT_DIR)

def clean_summary_file(file_path):
    """Clean a summary CSV file and save the cleaned version."""
    df = pd.read_csv(file_path)

    # Drop duplicate rows
    df = df.drop_duplicates()

    # Handle missing values (fill with 0 for now, can be changed later)
    df = df.fillna(0)

    # Flatten multi-index columns if needed
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = ['_'.join(col).strip() for col in df.columns.values]

    # Save cleaned summary file
    cleaned_file_path = os.path.join(CLEANED_OUTPUT_DIR, os.path.basename(file_path))
    df.to_csv(cleaned_file_path, index=False)

    print(f"Cleaned: {file_path} -> {cleaned_file_path}")

def clean_all_summaries():
    """Find and clean all summary files in the output directory."""
    for file_name in os.listdir(OUTPUT_DIR):
        if file_name.startswith("summary_") and file_name.endswith(".csv"):
            file_path = os.path.join(OUTPUT_DIR, file_name)
            clean_summary_file(file_path)

if __name__ == "__main__":
    clean_all_summaries()
