"""
Prepare and combine freight data for analysis
"""

import pandas as pd
import numpy as np
from pathlib import Path
import logging

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def combine_freight_data(data_dir: Path, output_dir: Path):
    """Combine freight data from multiple years"""
    try:
        all_data = []
        processed_years = set()
        
        # Read and combine data from 2020-2024
        for year in range(2020, 2025):
            year_dir = data_dir / str(year)
            if year_dir.exists() and year_dir.is_dir():
                logger.info(f"Processing data for year {year}")
                processed_years.add(year)
                months_processed = 0
                
                # Process each month directory
                for month_dir in year_dir.iterdir():
                    if month_dir.is_dir():
                        logger.info(f"Processing directory: {month_dir.name}")
                        months_processed += 1
                        files_processed = 0
                        
                        # Read only the monthly DOT files (exclude YTD files)
                        for dot_num in range(1, 4):  # DOT1, DOT2, DOT3
                            file_pattern = f"dot{dot_num}_*.csv"
                            monthly_files = list(month_dir.glob(file_pattern))
                            monthly_files = [f for f in monthly_files if 'ytd' not in f.name.lower()]
                            
                            for file_path in monthly_files:
                                logger.info(f"Reading file: {file_path.name}")
                                files_processed += 1
                                try:
                                    # Read CSV with specific data types and handle missing values
                                    if dot_num == 1:
                                        # DOT1 files
                                        df = pd.read_csv(file_path, dtype={
                                            'TRDTYPE': 'Int64',
                                            'USASTATE': 'str',
                                            'DEPE': 'str',
                                            'DISAGMOT': 'Int64',
                                            'MEXSTATE': 'str',
                                            'CANPROV': 'str',
                                            'COUNTRY': 'Int64',
                                            'VALUE': 'float64',
                                            'SHIPWT': 'float64',
                                            'FREIGHT_CHARGES': 'float64',
                                            'DF': 'float64',
                                            'CONTCODE': 'str',
                                            'MONTH': 'Int64',
                                            'YEAR': 'Int64'
                                        }, na_values=[''], keep_default_na=True)
                                    elif dot_num == 2:
                                        # DOT2 files have COMMODITY2 instead of DEPE
                                        df = pd.read_csv(file_path, dtype={
                                            'TRDTYPE': 'Int64',
                                            'USASTATE': 'str',
                                            'COMMODITY2': 'str',
                                            'DISAGMOT': 'Int64',
                                            'MEXSTATE': 'str',
                                            'CANPROV': 'str',
                                            'COUNTRY': 'Int64',
                                            'VALUE': 'float64',
                                            'SHIPWT': 'float64',
                                            'FREIGHT_CHARGES': 'float64',
                                            'DF': 'float64',
                                            'CONTCODE': 'str',
                                            'MONTH': 'Int64',
                                            'YEAR': 'Int64'
                                        }, na_values=[''], keep_default_na=True)
                                        # Rename COMMODITY2 to DEPE for consistency
                                        df = df.rename(columns={'COMMODITY2': 'DEPE'})
                                    else:
                                        # DOT3 files have different structure
                                        df = pd.read_csv(file_path, dtype={
                                            'TRDTYPE': 'Int64',
                                            'DEPE': 'str',
                                            'COMMODITY2': 'str',
                                            'DISAGMOT': 'Int64',
                                            'COUNTRY': 'Int64',
                                            'VALUE': 'float64',
                                            'SHIPWT': 'float64',
                                            'FREIGHT_CHARGES': 'float64',
                                            'DF': 'float64',
                                            'CONTCODE': 'str',
                                            'MONTH': 'Int64',
                                            'YEAR': 'Int64'
                                        }, na_values=[''], keep_default_na=True)
                                        # Add empty columns for consistency
                                        df['USASTATE'] = ''
                                        df['MEXSTATE'] = ''
                                        df['CANPROV'] = ''
                                    
                                    # Fill missing values
                                    df['USASTATE'] = df['USASTATE'].fillna('')
                                    df['DEPE'] = df['DEPE'].fillna('')
                                    df['MEXSTATE'] = df['MEXSTATE'].fillna('')
                                    df['CANPROV'] = df['CANPROV'].fillna('')
                                    df['CONTCODE'] = df['CONTCODE'].fillna('')
                                    
                                    # Add metadata
                                    df['Year'] = year
                                    month_name = month_dir.name.split('2')[0].strip()
                                    df['MonthName'] = month_name
                                    df['DOT_Type'] = f"DOT{dot_num}"
                                    
                                    # Extract month number from directory name
                                    month_map = {
                                        'Jan': 1, 'Feb': 2, 'Mar': 3, 'April': 4,
                                        'May': 5, 'June': 6, 'July': 7, 'August': 8,
                                        'September': 9, 'October': 10, 'November': 11,
                                        'December': 12
                                    }
                                    month_num = month_map.get(month_name, 1)  # Default to January if not found
                                    df['Month'] = month_num
                                    
                                    # Create date from Month and Year
                                    df['Date'] = pd.to_datetime(
                                        df['Year'].astype(str) + '-' + 
                                        df['Month'].astype(str).str.zfill(2) + '-01'
                                    )
                                    
                                    # Calculate value density (handle division by zero)
                                    df['ValueDensity'] = np.where(
                                        df['SHIPWT'] > 0,
                                        df['VALUE'] / df['SHIPWT'],
                                        0
                                    )
                                    
                                    all_data.append(df)
                                except Exception as e:
                                    logger.error(f"Error reading {file_path}: {str(e)}")
                                    raise
                        logger.info(f"Processed {files_processed} DOT files for {month_dir.name}")
                logger.info(f"Processed {months_processed} months for year {year}")
            else:
                logger.warning(f"No data directory found for year {year}")
        
        if not all_data:
            raise ValueError("No data files found")
            
        # Validate data coverage
        logger.info("\nData Coverage Summary:")
        logger.info(f"Years processed: {sorted(list(processed_years))}")
        missing_years = set(range(2020, 2025)) - processed_years
        if missing_years:
            logger.warning(f"Missing data for years: {sorted(list(missing_years))}")
        
        # Combine all data
        combined_df = pd.concat(all_data, ignore_index=True)
        
        # Add derived features
        combined_df['Season'] = pd.cut(combined_df['Month'], 
                                     bins=[0, 3, 6, 9, 12], 
                                     labels=['Winter', 'Spring', 'Summer', 'Fall'])
        
        # Save combined data
        output_file = output_dir / "freight_data_combined.csv"
        combined_df.to_csv(output_file, index=False)
        logger.info(f"Combined data saved to {output_file}")
        
        # Print summary statistics
        logger.info("\nData Summary:")
        logger.info(f"Total records: {len(combined_df):,}")
        logger.info(f"Date range: {combined_df['Date'].min()} to {combined_df['Date'].max()}")
        logger.info(f"Transport modes: {', '.join(map(str, combined_df['DISAGMOT'].unique()))}")
        logger.info(f"Total value: ${combined_df['VALUE'].sum():,.2f}")
        logger.info(f"Total weight: {combined_df['SHIPWT'].sum():,.2f}")
        logger.info("\nRecords by DOT type:")
        for dot_type in combined_df['DOT_Type'].unique():
            count = len(combined_df[combined_df['DOT_Type'] == dot_type])
            logger.info(f"{dot_type}: {count:,} records")
        
        return combined_df
        
    except Exception as e:
        logger.error(f"Error combining freight data: {str(e)}")
        raise

if __name__ == "__main__":
    # Set up directories
    base_dir = Path(__file__).parent.parent
    data_dir = base_dir / "data"
    output_dir = base_dir / "output"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Combine data
    combine_freight_data(data_dir, output_dir)
