import pandas as pd
import numpy as np
from pathlib import Path
import logging
from typing import Optional, Dict, Any
import glob

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class FreightDataPreprocessor:
    def __init__(self, data_dir: Path, output_dir: Path):
        """Initialize with data and output directories."""
        self.data_dir = Path(data_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
    def _process_single_file(self, file_path: Path) -> Optional[pd.DataFrame]:
        """Process a single CSV file."""
        try:
            logger.info(f"Processing file: {file_path.name}")
            
            # Try different encodings
            encodings = ['utf-8', 'latin1', 'cp1252']
            df = None
            
            for encoding in encodings:
                try:
                    df = pd.read_csv(file_path, encoding=encoding, low_memory=False)
                    break
                except UnicodeDecodeError:
                    continue
            
            if df is None:
                logger.error(f"Could not read {file_path.name} with any encoding")
                return None
            
            # Clean column names
            df.columns = df.columns.str.strip().str.upper()
            
            # Convert numeric columns
            numeric_cols = ['VALUE', 'SHIPWT', 'FREIGHT_CHARGES']
            for col in numeric_cols:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
            
            # Fill NaN values
            df['VALUE'] = df['VALUE'].fillna(0)
            df['SHIPWT'] = df['SHIPWT'].fillna(0)
            df['FREIGHT_CHARGES'] = df['FREIGHT_CHARGES'].fillna(0)
            
            # Add date column
            if 'MONTH' in df.columns and 'YEAR' in df.columns:
                df['Date'] = pd.to_datetime(df['YEAR'].astype(str) + '-' + df['MONTH'].astype(str).str.zfill(2) + '-01')
            
            return df
            
        except Exception as e:
            logger.error(f"Error processing {file_path.name}: {str(e)}")
            return None
    
    def _add_derived_metrics(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add derived metrics to the DataFrame."""
        try:
            # Calculate value density (value per weight)
            mask = (df['SHIPWT'] > 0) & (df['VALUE'] > 0)
            df.loc[mask, 'value_density'] = df.loc[mask, 'VALUE'] / df.loc[mask, 'SHIPWT']
            
            # Calculate cost efficiency (freight charges per value)
            mask = (df['VALUE'] > 0) & (df['FREIGHT_CHARGES'] > 0)
            df.loc[mask, 'cost_per_value'] = df.loc[mask, 'FREIGHT_CHARGES'] / df.loc[mask, 'VALUE']
            
            return df
            
        except Exception as e:
            logger.error(f"Error adding derived metrics: {str(e)}")
            return df
    
    def process_year_data(self, year: str) -> Optional[pd.DataFrame]:
        """Process all data files for a given year."""
        try:
            logger.info(f"Processing data for year {year}")
            
            # Get all CSV files for the year
            year_dir = self.data_dir / year
            all_dfs = []
            
            # Process each month
            for month in range(1, 13):
                month_dir = year_dir / f"{month:02d}"
                if not month_dir.exists():
                    continue
                    
                for file_path in month_dir.glob('*.csv'):
                    df = self._process_single_file(file_path)
                    if df is not None:
                        all_dfs.append(df)
            
            if not all_dfs:
                logger.warning(f"No data found for year {year}")
                return None
            
            # Combine all DataFrames
            combined_df = pd.concat(all_dfs, ignore_index=True)
            
            # Add derived metrics
            combined_df = self._add_derived_metrics(combined_df)
            
            # Save to parquet
            output_file = self.output_dir / f'freight_data_{year}_processed.parquet'
            combined_df.to_parquet(output_file, index=False)
            logger.info(f"Saved processed data to {output_file}")
            
            return combined_df
            
        except Exception as e:
            logger.error(f"Error processing year {year}: {str(e)}")
            return None

def main():
    """Main function to run the data preparation."""
    base_dir = Path(__file__).parent.parent
    processor = FreightDataPreprocessor(
        data_dir=base_dir / 'data',
        output_dir=base_dir / 'output'
    )
    
    # Process each year
    for year in range(2020, 2025):
        processor.process_year_data(str(year))

if __name__ == "__main__":
    main()
