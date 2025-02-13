import pandas as pd
import numpy as np
from pathlib import Path
import logging
from typing import Optional, Dict, Any
import glob
import json

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
        
        # Define standard column mappings
        self.column_mappings = {
            'DISAGMOT': 'transport_mode',
            'VALUE': 'shipment_value',
            'SHIPWT': 'shipment_weight',
            'FREIGHT_CHARGES': 'freight_charges',
            'USASTATE': 'origin_state',
            'MEXSTATE': 'mexico_state',
            'CANPROV': 'canada_province'
        }
        
        # Initialize data quality metrics
        self.quality_metrics = {
            'missing_values': {},
            'outliers': {},
            'invalid_dates': [],
            'invalid_values': []
        }
    
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
            
            # Clean and standardize the dataset
            df = self.clean_data(df)
            
            return df
            
        except Exception as e:
            logger.error(f"Error processing {file_path.name}: {str(e)}")
            return None
    
    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean and standardize the dataset."""
        try:
            # Make a copy to avoid modifying original
            df = df.copy()
            
            # Standardize column names
            df.columns = df.columns.str.strip().str.upper()
            
            # Handle missing values
            for col in df.columns:
                missing_count = df[col].isna().sum()
                if missing_count > 0:
                    self.quality_metrics['missing_values'][col] = missing_count
                    
                    if col in ['VALUE', 'SHIPWT', 'FREIGHT_CHARGES']:
                        # For numeric columns, interpolate missing values
                        df[col] = df[col].interpolate(method='linear')
                    else:
                        # For categorical columns, fill with mode
                        df[col] = df[col].fillna(df[col].mode()[0])
            
            # Convert numeric columns
            numeric_cols = ['VALUE', 'SHIPWT', 'FREIGHT_CHARGES']
            for col in numeric_cols:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
                    
                    # Detect and handle outliers
                    Q1 = df[col].quantile(0.25)
                    Q3 = df[col].quantile(0.75)
                    IQR = Q3 - Q1
                    lower_bound = Q1 - 1.5 * IQR
                    upper_bound = Q3 + 1.5 * IQR
                    
                    outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)][col]
                    if not outliers.empty:
                        self.quality_metrics['outliers'][col] = len(outliers)
                        
                        # Cap outliers at bounds
                        df[col] = df[col].clip(lower=lower_bound, upper=upper_bound)
            
            # Standardize dates
            if 'MONTH' in df.columns and 'YEAR' in df.columns:
                df['Date'] = pd.to_datetime(df['YEAR'].astype(str) + '-' + df['MONTH'].astype(str).str.zfill(2) + '-01')
                invalid_dates = df['Date'].isna().sum()
                if invalid_dates > 0:
                    self.quality_metrics['invalid_dates'].append(invalid_dates)
            
            # Add derived features
            df['value_density'] = df['VALUE'] / df['SHIPWT']
            df['cost_per_value'] = df['FREIGHT_CHARGES'] / df['VALUE']
            
            # Validate value ranges
            df = self._validate_values(df)
            
            return df
            
        except Exception as e:
            logger.error(f"Error cleaning data: {str(e)}")
            raise
    
    def _validate_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """Validate value ranges and flag invalid entries."""
        try:
            # Value must be positive
            invalid_values = df['VALUE'] <= 0
            if invalid_values.any():
                self.quality_metrics['invalid_values'].append(
                    f"Found {invalid_values.sum()} non-positive values in VALUE column"
                )
                df.loc[invalid_values, 'VALUE'] = df['VALUE'].median()
            
            # Weight must be positive
            invalid_weights = df['SHIPWT'] <= 0
            if invalid_weights.any():
                self.quality_metrics['invalid_values'].append(
                    f"Found {invalid_weights.sum()} non-positive values in SHIPWT column"
                )
                df.loc[invalid_weights, 'SHIPWT'] = df['SHIPWT'].median()
            
            # Freight charges should be positive
            invalid_charges = df['FREIGHT_CHARGES'] < 0
            if invalid_charges.any():
                self.quality_metrics['invalid_values'].append(
                    f"Found {invalid_charges.sum()} negative values in FREIGHT_CHARGES column"
                )
                df.loc[invalid_charges, 'FREIGHT_CHARGES'] = 0
            
            return df
            
        except Exception as e:
            logger.error(f"Error validating values: {str(e)}")
            raise
    
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
            
            # Save to parquet
            output_file = self.output_dir / f'freight_data_{year}_processed.parquet'
            combined_df.to_parquet(output_file, index=False)
            logger.info(f"Saved processed data to {output_file}")
            
            # Save data quality report
            self.save_quality_report()
            
            return combined_df
            
        except Exception as e:
            logger.error(f"Error processing year {year}: {str(e)}")
            return None
    
    def save_quality_report(self):
        """Save data quality metrics to a JSON file."""
        try:
            report_path = self.output_dir / 'data_quality_report.json'
            with open(report_path, 'w') as f:
                json.dump(self.quality_metrics, f, indent=2)
            logger.info(f"Saved data quality report to {report_path}")
        except Exception as e:
            logger.error(f"Error saving quality report: {str(e)}")

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
