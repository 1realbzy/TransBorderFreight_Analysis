# Development Guidelines & Improvement Recommendations

## Quick Fix Recommendations

### Critical Issues (Fix Immediately)

#### 1. Path Configuration Issue
**Problem**: Hardcoded Windows path prevents cross-platform execution
```python
# Line 17 in process_freight_data.py
self.base_dir = Path(r"c:\Users\hbempong\TransBorderFreight_Analysis\data")
```

**Solution**: Use relative paths or environment variables
```python
# Recommended fix
import os
self.base_dir = Path(os.getenv('FREIGHT_DATA_DIR', './data'))
self.output_dir = Path(os.getenv('FREIGHT_OUTPUT_DIR', './output'))
```

#### 2. Improve Error Handling
**Current Issue**: Generic exception handling may mask specific errors
```python
# Current pattern
except Exception as e:
    logger.error(f"Error: {str(e)}")
```

**Recommended Enhancement**:
```python
# More specific error handling
except FileNotFoundError as e:
    logger.error(f"Data file not found: {e}")
    return pd.DataFrame()
except pd.errors.EmptyDataError as e:
    logger.error(f"Empty data file: {e}")
    continue
except Exception as e:
    logger.error(f"Unexpected error: {e}\n{traceback.format_exc()}")
    raise
```

### Medium Priority Improvements

#### 1. Add Configuration Management
Create a configuration file system:

```python
# config.py
import json
from pathlib import Path

class Config:
    def __init__(self, config_file='config.json'):
        self.load_config(config_file)
    
    def load_config(self, config_file):
        default_config = {
            'data_dir': './data',
            'output_dir': './output',
            'years': [2020, 2021, 2022, 2023, 2024],
            'emission_factors': {
                'RAIL': 0.023,
                'ROAD': 0.161,
                # ... etc
            }
        }
        
        if Path(config_file).exists():
            with open(config_file, 'r') as f:
                user_config = json.load(f)
            default_config.update(user_config)
        
        self.__dict__.update(default_config)
```

#### 2. Add Input Validation
```python
def validate_dataframe(self, df: pd.DataFrame) -> bool:
    """Validate input DataFrame has required columns and data types."""
    required_columns = ['DISAGMOT', 'VALUE', 'SHIPWT']
    
    missing_cols = [col for col in required_columns if col not in df.columns]
    if missing_cols:
        logger.error(f"Missing required columns: {missing_cols}")
        return False
    
    # Validate data types
    if not pd.api.types.is_numeric_dtype(df['VALUE']):
        logger.warning("VALUE column is not numeric, attempting conversion")
        df['VALUE'] = pd.to_numeric(df['VALUE'], errors='coerce')
    
    return True
```

#### 3. Performance Optimization
```python
def process_csv_files_optimized(self, year: int) -> pd.DataFrame:
    """Optimized version with chunking for large files."""
    extract_dir = self.base_dir / str(year) / "extracted"
    if not extract_dir.exists():
        logger.error(f"No extracted data found for {year}")
        return pd.DataFrame()
    
    csv_files = list(extract_dir.glob("**/*.csv"))
    logger.info(f"Found {len(csv_files)} CSV files for year {year}")
    
    # Process files in chunks to manage memory
    chunk_size = 10000
    dfs = []
    
    for csv_file in csv_files:
        try:
            # Read in chunks for large files
            for chunk in pd.read_csv(csv_file, chunksize=chunk_size, low_memory=False):
                processed_chunk = self.process_chunk(chunk, year)
                if not processed_chunk.empty:
                    dfs.append(processed_chunk)
        except Exception as e:
            logger.error(f"Error processing {csv_file}: {str(e)}")
            continue
    
    return pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()
```

## Testing Strategy

### 1. Unit Tests Structure
```python
# tests/test_freight_processor.py
import unittest
import pandas as pd
from scripts.process_freight_data import FreightDataProcessor

class TestFreightDataProcessor(unittest.TestCase):
    def setUp(self):
        self.processor = FreightDataProcessor()
        
        # Create sample test data
        self.sample_data = pd.DataFrame({
            'DISAGMOT': [1, 3, 5],
            'VALUE': [1000, 2000, 3000],
            'SHIPWT': [100, 200, 300],
            'MONTH': [1, 2, 3],
            'YEAR': [2020, 2020, 2020]
        })
    
    def test_mode_mapping(self):
        self.assertEqual(self.processor.mode_mapping[1], 'RAIL')
        self.assertEqual(self.processor.mode_mapping[5], 'VESSEL')
    
    def test_analyze_data_basic(self):
        result = self.processor.analyze_data(self.sample_data)
        self.assertIn('mode_analysis', result)
        self.assertIn('efficiency', result)
        self.assertIn('environmental_impact', result)
    
    def test_empty_dataframe_handling(self):
        empty_df = pd.DataFrame()
        result = self.processor.analyze_data(empty_df)
        self.assertEqual(result, {})
    
    def test_insight_generation(self):
        mock_analysis = {
            2020: {
                'mode_analysis': {
                    'modal_split': {'RAIL': 10.0, 'VESSEL': 60.0}
                },
                'efficiency': {'RAIL': {'mean': 100.0}},
                'environmental_impact': {'RAIL': {'emissions_per_ton': 0.023}},
                'seasonal_patterns': {1: {'RAIL': {'value': 1000}}},
                'trade_corridors': {'TX': {'RAIL': {'value': 1000}}}
            }
        }
        insights = self.processor.generate_insights(mock_analysis)
        self.assertIn('business_question_answers', insights)
        self.assertIn('recommendations', insights)

if __name__ == '__main__':
    unittest.main()
```

### 2. Integration Tests
```python
# tests/test_integration.py
import tempfile
import shutil
from pathlib import Path

class TestIntegration(unittest.TestCase):
    def setUp(self):
        # Create temporary test environment
        self.test_dir = tempfile.mkdtemp()
        self.data_dir = Path(self.test_dir) / 'data'
        self.output_dir = Path(self.test_dir) / 'output'
        
        # Create sample test data files
        self.create_sample_data()
    
    def tearDown(self):
        shutil.rmtree(self.test_dir)
    
    def create_sample_data(self):
        # Create sample CSV with test data
        pass
    
    def test_end_to_end_pipeline(self):
        # Test complete processing pipeline
        pass
```

### 3. Data Quality Tests
```python
def test_data_quality_checks(self):
    """Test data validation and quality checks."""
    # Test handling of missing values
    # Test data type conversions
    # Test outlier detection
    pass
```

## Code Quality Improvements

### 1. Type Hints Enhancement
```python
from typing import Dict, List, Any, Optional, Union
import pandas as pd

class FreightDataProcessor:
    def __init__(self, config_path: Optional[str] = None) -> None:
        # Implementation
        pass
    
    def process_csv_files(self, year: int) -> pd.DataFrame:
        # Implementation
        pass
    
    def analyze_data(self, df: pd.DataFrame) -> Dict[str, Any]:
        # Implementation
        pass
    
    def generate_insights(self, 
                         all_years_analysis: Dict[int, Dict[str, Any]]
                         ) -> Dict[str, List[str]]:
        # Implementation
        pass
```

### 2. Documentation Enhancement
```python
def analyze_data(self, df: pd.DataFrame) -> Dict[str, Any]:
    """
    Perform comprehensive analysis on freight data.
    
    This method conducts multi-dimensional analysis including transport mode
    distribution, efficiency metrics, environmental impact assessment,
    seasonal patterns, and trade corridor analysis.
    
    Args:
        df (pd.DataFrame): Preprocessed freight data containing required columns:
            - DISAGMOT: Transport mode identifier (1-9)
            - VALUE: Freight value in USD
            - SHIPWT: Shipment weight in tons
            - Additional optional columns for geographic/temporal analysis
    
    Returns:
        Dict[str, Any]: Analysis results with the following structure:
            - mode_analysis: Modal distribution and metrics
            - efficiency: Value-per-weight analysis by mode
            - environmental_impact: CO2 emissions by transport mode
            - seasonal_patterns: Monthly freight distribution
            - trade_corridors: Geographic origin-destination analysis
    
    Raises:
        ValueError: If DataFrame is empty or missing required columns
        KeyError: If transport mode mapping fails
    
    Example:
        >>> processor = FreightDataProcessor()
        >>> df = pd.read_csv('freight_data.csv')
        >>> results = processor.analyze_data(df)
        >>> print(results['mode_analysis']['modal_split'])
        {'VESSEL': 65.0, 'RAIL': 7.5, 'ROAD': 4.3, ...}
    """
```

### 3. Logging Enhancement
```python
import logging
from functools import wraps

def log_method_call(func):
    """Decorator to log method entry and exit."""
    @wraps(func)
    def wrapper(self, *args, **kwargs):
        logger.info(f"Starting {func.__name__} with args: {args[:2]}")  # Limit args for readability
        try:
            result = func(self, *args, **kwargs)
            logger.info(f"Completed {func.__name__} successfully")
            return result
        except Exception as e:
            logger.error(f"Error in {func.__name__}: {str(e)}")
            raise
    return wrapper

class FreightDataProcessor:
    @log_method_call
    def process_csv_files(self, year: int) -> pd.DataFrame:
        # Implementation
        pass
```

## Security & Best Practices

### 1. Input Sanitization
```python
def sanitize_year_input(self, year: Union[int, str]) -> int:
    """Sanitize and validate year input."""
    try:
        year_int = int(year)
        if not (2000 <= year_int <= 2030):
            raise ValueError(f"Year {year_int} outside valid range (2000-2030)")
        return year_int
    except (ValueError, TypeError) as e:
        logger.error(f"Invalid year input: {year}")
        raise ValueError(f"Invalid year: {year}") from e
```

### 2. Path Security
```python
def validate_path(self, path: Path) -> Path:
    """Validate path is within allowed directories."""
    try:
        resolved_path = path.resolve()
        # Ensure path is within project directory
        if not str(resolved_path).startswith(str(Path.cwd().resolve())):
            raise ValueError("Path outside allowed directory")
        return resolved_path
    except Exception as e:
        logger.error(f"Invalid path: {path}")
        raise
```

## Performance Monitoring

### 1. Add Performance Metrics
```python
import time
from functools import wraps

def measure_performance(func):
    """Decorator to measure method execution time."""
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        logger.info(f"{func.__name__} executed in {end_time - start_time:.2f} seconds")
        return result
    return wrapper

@measure_performance
def process_all_years(self):
    # Implementation
    pass
```

### 2. Memory Usage Monitoring
```python
import psutil
import os

def log_memory_usage(stage: str):
    """Log current memory usage."""
    process = psutil.Process(os.getpid())
    memory_mb = process.memory_info().rss / 1024 / 1024
    logger.info(f"Memory usage at {stage}: {memory_mb:.1f} MB")
```

## Deployment Improvements

### 1. Environment Configuration
```bash
# .env file
FREIGHT_DATA_DIR=./data
FREIGHT_OUTPUT_DIR=./output
LOG_LEVEL=INFO
MAX_WORKERS=4
CHUNK_SIZE=10000
```

### 2. Docker Support
```dockerfile
# Dockerfile
FROM python:3.9-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY scripts/ ./scripts/
COPY data/ ./data/

CMD ["python", "scripts/process_freight_data.py"]
```

### 3. CI/CD Pipeline
```yaml
# .github/workflows/test.yml
name: Test
on: [push, pull_request]
jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - name: Set up Python
        uses: actions/setup-python@v2
        with:
          python-version: 3.9
      - name: Install dependencies
        run: pip install -r requirements.txt
      - name: Run tests
        run: python -m pytest tests/
```

## Implementation Priority

### Phase 1 (Immediate - 1-2 days)
1. ✅ Fix hardcoded path issue
2. ✅ Add basic input validation
3. ✅ Improve error handling specificity
4. ✅ Add configuration management

### Phase 2 (Short-term - 1 week)
1. ✅ Create comprehensive test suite
2. ✅ Add performance monitoring
3. ✅ Enhance documentation
4. ✅ Implement logging improvements

### Phase 3 (Medium-term - 2-4 weeks)
1. ✅ Add parallel processing capabilities
2. ✅ Implement data caching
3. ✅ Create REST API interface
4. ✅ Add interactive dashboard

### Phase 4 (Long-term - 1-3 months)
1. ✅ Add predictive analytics
2. ✅ Implement real-time processing
3. ✅ Create web-based interface
4. ✅ Add machine learning capabilities

This phased approach ensures incremental improvements while maintaining system stability and functionality.