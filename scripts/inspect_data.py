"""
Data Inspection Script for TransBorder Freight Analysis
"""
import pandas as pd
from pathlib import Path

def inspect_sample_file():
    # Get path to a sample CSV file
    data_dir = Path(__file__).parent.parent / "data" / "2020" / "April2020" / "Apr 2020"
    sample_file = data_dir / "dot1_0420.csv"
    
    # Read with all columns as string first to avoid mixed type issues
    df = pd.read_csv(sample_file, dtype=str)
    
    print("File Path:", sample_file)
    print("\nFirst few rows:")
    print(df.head())
    
    print("\nColumns and their unique values:")
    for col in df.columns:
        n_unique = len(df[col].unique())
        print(f"\n{col}:")
        print(f"Number of unique values: {n_unique}")
        if n_unique < 10:  # Only show unique values if there aren't too many
            print("Unique values:", df[col].unique())
            
    return df

if __name__ == "__main__":
    df = inspect_sample_file()
