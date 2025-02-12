import os
import shutil
from pathlib import Path
import logging

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def organize_data_files():
    """Organize data files into a consistent structure."""
    base_dir = Path(__file__).parent.parent
    data_dir = base_dir / 'data'
    
    # Create year directories if they don't exist
    for year in range(2020, 2025):
        year_dir = data_dir / str(year)
        year_dir.mkdir(exist_ok=True)
        
        # Create month directories
        for month in range(1, 13):
            month_dir = year_dir / f"{month:02d}"
            month_dir.mkdir(exist_ok=True)
    
    # Move files to appropriate directories
    for root, _, files in os.walk(data_dir):
        root_path = Path(root)
        
        for file in files:
            if not file.endswith('.csv'):
                continue
                
            file_path = root_path / file
            
            # Extract year and month from directory name or file name
            year = None
            month = None
            
            # Try to get year from parent directory
            for parent in file_path.parents:
                if parent.name.isdigit() and len(parent.name) == 4:
                    year = parent.name
                    break
            
            # Try to get month from file name
            if '_' in file:
                month_str = file.split('_')[1][:2]
                if month_str.isdigit():
                    month = int(month_str)
            
            if year and month:
                # Create target directory path
                target_dir = data_dir / year / f"{month:02d}"
                target_dir.mkdir(exist_ok=True)
                
                # Move file to target directory
                target_path = target_dir / file
                if not target_path.exists():
                    try:
                        shutil.move(str(file_path), str(target_path))
                        logger.info(f"Moved {file} to {target_path}")
                    except Exception as e:
                        logger.error(f"Error moving {file}: {str(e)}")

if __name__ == "__main__":
    organize_data_files()
