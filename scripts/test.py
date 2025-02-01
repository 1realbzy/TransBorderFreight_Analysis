# scripts/consolidate_data.py
import os
import pandas as pd
import glob

DATA_DIR = r"C:\Users\hbempong\TransBorderFreight_Analysis\data\2020"
OUTPUT_FILE = r"C:\Users\hbempong\TransBorderFreight_Analysis\output\merged_data.csv"

# Mapping dictionary for human-readable labels
MAPPINGS = {
    "DISAGMOT": {
        5: "Truck", 6: "Rail", 7: "Pipeline", 8: "Air", 
        9: "Vessel", 10: "Other"
    },
    "CANPROV": {
        "XA": "Alberta", "XB": "British Columbia", "XC": "Manitoba", 
        "XD": "New Brunswick", "XE": "Newfoundland and Labrador", 
        "XF": "Nova Scotia", "XG": "Ontario", "XH": "Prince Edward Island",
        "XI": "Quebec", "XJ": "Saskatchewan", "XO": "Other",
        "XX": "Unknown"
    },
    "MEXSTATE": {
        "XX": "Unknown", "XO": "Other", "XM": "Mexico City", 
        "XQ": "Querétaro", "XY": "Yucatán", "XA": "Aguascalientes",
        "XB": "Baja California", "XC": "Campeche", "XD": "Chiapas",
        "XE": "Chihuahua", "XF": "Coahuila", "XG": "Colima",
        "XH": "Durango", "XI": "Guanajuato", "XJ": "Guerrero",
        "XK": "Hidalgo", "XL": "Jalisco", "XN": "Michoacán",
        "XP": "Nayarit", "XR": "Oaxaca", "XS": "Puebla",
        "XT": "Querétaro", "XU": "Quintana Roo", "XV": "San Luis Potosí",
        "XW": "Sinaloa", "XZ": "Tamaulipas", "YA": "Tlaxcala", 
        "YB": "Veracruz", "YC": "Yucatán", "YD": "Zacatecas"
    }
}

def load_and_merge_data(data_dir):
    """Load and merge all monthly CSV files into one DataFrame."""
    all_files = glob.glob(os.path.join(data_dir, "*/*.csv"))

    dtype_spec = {
        "DISAGMOT": "Int64",
        "CANPROV": "string",
        "VALUE": "float64",
        "SHIPWT": "float64",
        "DEPE": "string"
    }

    df_list = []
    for file in all_files:
        try:
            # Process files in chunks to save memory
            chunk_iter = pd.read_csv(file, dtype=dtype_spec, low_memory=False, chunksize=100000)
            for chunk in chunk_iter:
                df_list.append(chunk)
        except Exception as e:
            print(f"Error reading {file}: {e}")

    df = pd.concat(df_list, ignore_index=True)
    print(f"Loaded {len(df)} rows from {len(all_files)} files.")
    return df

def infer_missing_values(df):
    """Infer missing values for CANPROV and MEXSTATE where possible."""
    if "COUNTRY" not in df.columns:
        raise ValueError("Column 'COUNTRY' not found in the dataset.")

    # Fill missing CANPROV based on COUNTRY
    df.loc[(df["CANPROV"].isna()) & (df["COUNTRY"] == 1220), "CANPROV"] = "XO"  # Assign "Other" for Canada
    
    # Fill missing MEXSTATE based on COUNTRY
    df.loc[(df["MEXSTATE"].isna()) & (df["COUNTRY"] == 2010), "MEXSTATE"] = "XO"  # Assign "Other" for Mexico

    return df

def clean_data(df):
    """Clean and preprocess the dataset."""
    df = infer_missing_values(df)

    # Replace codes with human-readable labels
    for column, mapping in MAPPINGS.items():
        if column in df.columns:
            df[column] = df[column].map(mapping).fillna("Unknown")

    # Log missing values for review
    for column in MAPPINGS.keys():
        if column in df.columns:
            missing_values = df[~df[column].isin(MAPPINGS[column].values())][column].unique()
            if len(missing_values) > 0:
                print(f"Warning: Found unmapped values in {column}: {missing_values}")

    # Fix SHIPWT = 0 cases
    df.loc[(df["SHIPWT"] == 0) & (df["VALUE"] > 0), "SHIPWT"] = df["SHIPWT"].replace(0, df["SHIPWT"].median())

    # Drop rows where both SHIPWT and VALUE are 0
    df = df[(df["SHIPWT"] > 0) & (df["VALUE"] > 0)]

    # Feature Engineering: Calculate Freight Density
    df["FREIGHT_DENSITY"] = df["VALUE"] / df["SHIPWT"]

    return df

def main():
    df = load_and_merge_data(DATA_DIR)
    df_cleaned = clean_data(df)
    
    # Save the cleaned dataset
    df_cleaned.to_csv(OUTPUT_FILE, index=False)
    print(f"Cleaned data saved to {OUTPUT_FILE}")

if __name__ == "__main__":
    main()