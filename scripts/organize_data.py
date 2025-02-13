import os
import shutil
from pathlib import Path
import logging
import json

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
    raw_dir = data_dir / 'raw'
    processed_dir = data_dir / 'processed'
    metadata_dir = data_dir / 'metadata'
    
    # Create necessary directories
    for directory in [raw_dir, processed_dir, metadata_dir]:
        directory.mkdir(exist_ok=True)
    
    # Create year directories
    for year in range(2020, 2025):
        year_dir = processed_dir / str(year)
        year_dir.mkdir(exist_ok=True)
        
        # Create month directories
        for month in range(1, 13):
            month_dir = year_dir / f"{month:02d}"
            month_dir.mkdir(exist_ok=True)
    
    # Track file metadata
    metadata = {
        'files': [],
        'total_size': 0,
        'file_count': 0,
        'years_covered': set(),
        'modes_present': set()
    }
    
    # Move and organize files
    for file_path in raw_dir.glob('*.csv'):
        try:
            # Extract date information
            date_info = extract_date_from_filename(file_path.name)
            if not date_info:
                logger.warning(f"Could not extract date from {file_path.name}")
                continue
            
            year, month = date_info
            
            # Validate year range
            if not (2020 <= int(year) <= 2024):
                logger.warning(f"File {file_path.name} has invalid year {year}")
                continue
            
            # Create target directory path
            target_dir = processed_dir / str(year) / f"{month:02d}"
            target_dir.mkdir(exist_ok=True)
            
            # Copy file to processed directory
            target_path = target_dir / file_path.name
            if not target_path.exists():
                shutil.copy2(file_path, target_path)
                
                # Update metadata
                file_stats = file_path.stat()
                metadata['files'].append({
                    'filename': file_path.name,
                    'year': year,
                    'month': month,
                    'size': file_stats.st_size,
                    'modified': file_stats.st_mtime
                })
                metadata['total_size'] += file_stats.st_size
                metadata['file_count'] += 1
                metadata['years_covered'].add(year)
                
                logger.info(f"Processed {file_path.name} -> {target_path}")
            
        except Exception as e:
            logger.error(f"Error processing {file_path.name}: {str(e)}")
    
    # Save metadata
    save_metadata(metadata, metadata_dir / 'data_inventory.json')
    
    return metadata

def extract_date_from_filename(filename: str) -> tuple:
    """Extract year and month from filename."""
    try:
        # Assuming filename format: freight_YYYYMM_additional_info.csv
        parts = filename.split('_')
        if len(parts) >= 2:
            date_part = parts[1]
            if len(date_part) >= 6 and date_part.isdigit():
                year = date_part[:4]
                month = date_part[4:6]
                return year, int(month)
    except Exception as e:
        logger.error(f"Error extracting date from {filename}: {str(e)}")
    
    return None

def save_metadata(metadata: dict, output_path: Path):
    """Save metadata to JSON file."""
    try:
        # Convert sets to lists for JSON serialization
        metadata['years_covered'] = list(metadata['years_covered'])
        metadata['modes_present'] = list(metadata['modes_present'])
        
        with open(output_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        logger.info(f"Saved metadata to {output_path}")
    except Exception as e:
        logger.error(f"Error saving metadata: {str(e)}")

def validate_data_organization():
    """Validate the data organization structure."""
    base_dir = Path(__file__).parent.parent
    data_dir = base_dir / 'data'
    processed_dir = data_dir / 'processed'
    
    validation_results = {
        'missing_months': [],
        'empty_directories': [],
        'file_count_by_year': {},
        'total_files': 0
    }
    
    # Check each year directory
    for year in range(2020, 2025):
        year_dir = processed_dir / str(year)
        if not year_dir.exists():
            continue
        
        file_count = 0
        
        # Check each month directory
        for month in range(1, 13):
            month_dir = year_dir / f"{month:02d}"
            if not month_dir.exists():
                validation_results['missing_months'].append(f"{year}-{month:02d}")
                continue
            
            # Count files in month directory
            month_files = list(month_dir.glob('*.csv'))
            file_count += len(month_files)
            
            if len(month_files) == 0:
                validation_results['empty_directories'].append(str(month_dir))
        
        validation_results['file_count_by_year'][year] = file_count
        validation_results['total_files'] += file_count
    
    return validation_results

if __name__ == "__main__":
    # Organize data files
    metadata = organize_data_files()
    
    # Validate organization
    validation_results = validate_data_organization()
    
    # Log results
    logger.info("Data Organization Complete")
    logger.info(f"Total files processed: {metadata['file_count']}")
    logger.info(f"Years covered: {', '.join(metadata['years_covered'])}")
    
    if validation_results['missing_months']:
        logger.warning(f"Missing months: {', '.join(validation_results['missing_months'])}")
    if validation_results['empty_directories']:
        logger.warning(f"Empty directories: {', '.join(validation_results['empty_directories'])}")
