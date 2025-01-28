# scripts/process_all_months.py
import os
import pandas as pd

# Base directory containing all monthly folders
DATA_BASE_DIR = r"C:\Users\hbempong\TransBorderFreight_Analysis\data\2020"
OUTPUT_DIR = r"C:\Users\hbempong\TransBorderFreight_Analysis\output"

# Ensure output directory exists
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

def process_csv_in_chunks(file_path, output_file, chunk_size=50000):
    """Process a large CSV in chunks, computing summaries and saving a sample."""
    summary_data = []
    sample_data = None

    for chunk in pd.read_csv(file_path, chunksize=chunk_size):
        summary_data.append(chunk.describe())  # Summarize numeric columns
        
        # Save first few rows as a sample (only once)
        if sample_data is None:
            sample_data = chunk.head(500)  

    # Save summary statistics
    summary_df = pd.concat(summary_data)
    summary_df.to_csv(output_file, index=False)
    
    # Save sample data for review
    sample_output_file = output_file.replace("summary_", "sample_")
    if sample_data is not None:
        sample_data.to_csv(sample_output_file, index=False)

    print(f"Processed {file_path}. Summary saved to {output_file}.")

def process_all_months():
    """Loop through all month folders and process CSV files."""
    for month_folder in os.listdir(DATA_BASE_DIR):
        month_path = os.path.join(DATA_BASE_DIR, month_folder)

        if os.path.isdir(month_path):  # Ensure it's a folder
            print(f"Processing data for {month_folder}...")

            for file_name in os.listdir(month_path):
                if file_name.endswith(".csv"):
                    file_path = os.path.join(month_path, file_name)
                    output_file = os.path.join(OUTPUT_DIR, f"summary_{month_folder}_{file_name}")
                    
                    # Process CSV in chunks
                    process_csv_in_chunks(file_path, output_file)

if __name__ == "__main__":
    process_all_months()
