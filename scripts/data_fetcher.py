"""
Data fetcher for TransBorder Freight Data
Downloads missing data files from the official source
"""

import requests
import pandas as pd
from pathlib import Path
import logging
from typing import List, Dict
from datetime import datetime
import os

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class TransBorderDataFetcher:
    def __init__(self, base_dir: Path):
        self.base_dir = base_dir
        self.data_dir = base_dir / "data"
        # Update URL pattern to match BTS website structure
        self.base_url = "https://www.bts.gov/browse-statistical-products-and-data/transborder-freight-data"
        
    def get_missing_periods(self) -> Dict[str, List]:
        """Get list of missing data periods"""
        from prepare_data import validate_data_coverage
        
        coverage = validate_data_coverage(self.data_dir)
        return {
            "missing": coverage["missing_periods"],
            "partial": coverage["partial_periods"]
        }
    
    def get_download_url(self, year: int, month: int, dot_type: int) -> str:
        """Construct the download URL for a specific file"""
        month_str = f"{month:02d}"
        year_str = str(year)[2:]
        
        # Example URL pattern (you'll need to verify this):
        # https://www.bts.gov/sites/bts.dot.gov/files/docs/browse-statistical-products-and-data/transborder-freight-data/220316/dot1_0122.csv
        return f"{self.base_url}/dot{dot_type}_{month_str}{year_str}.csv"
    
    def download_dot_file(self, year: int, month: int, dot_type: int) -> bool:
        """
        Download a specific DOT file
        Returns True if successful, False otherwise
        """
        try:
            url = self.get_download_url(year, month, dot_type)
            
            # Create year directory if it doesn't exist
            year_dir = self.data_dir / str(year)
            month_dir = year_dir / f"{datetime.strptime(f'{month:02d}', '%m').strftime('%B')} {year}"
            month_dir.mkdir(parents=True, exist_ok=True)
            
            # Download file
            response = requests.get(url)
            if response.status_code == 200:
                filename = f"dot{dot_type}_{f'{month:02d}'}{str(year)[2:]}.csv"
                file_path = month_dir / filename
                with open(file_path, 'wb') as f:
                    f.write(response.content)
                logger.info(f"Successfully downloaded {filename}")
                return True
            else:
                logger.warning(f"Failed to download {filename}: Status code {response.status_code}")
                return False
                
        except Exception as e:
            logger.error(f"Error downloading file: {str(e)}")
            return False
    
    def fetch_missing_data(self):
        """Fetch all missing data files"""
        missing_data = self.get_missing_periods()
        
        # Process completely missing periods
        for period in missing_data["missing"]:
            if "Complete year missing" in period:
                year = int(period.split()[1])
            else:
                year = int(period.split('-')[0])
                month = int(period.split('-')[1])
                
                # Try to download all DOT files for this period
                for dot_type in range(1, 4):
                    self.download_dot_file(year, month, dot_type)
        
        # Process partially missing periods
        for partial in missing_data["partial"]:
            period = partial["period"]
            year = int(period.split('-')[0])
            month = int(period.split('-')[1])
            
            # Download missing DOT files
            for missing_dot in partial["missing"]:
                dot_type = int(missing_dot.replace("DOT", ""))
                self.download_dot_file(year, month, dot_type)

if __name__ == "__main__":
    base_dir = Path(__file__).parent.parent
    fetcher = TransBorderDataFetcher(base_dir)
    
    logger.info("Starting data fetch process...")
    fetcher.fetch_missing_data()
    logger.info("Data fetch process completed")
