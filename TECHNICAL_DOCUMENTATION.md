# Technical Documentation - TransBorderFreight_Analysis

## API Reference

### FreightDataProcessor Class

#### Constructor
```python
FreightDataProcessor()
```
Initializes the freight data processor with default configuration.

**Attributes:**
- `base_dir`: Path to data directory (currently hardcoded)
- `output_dir`: Path to output directory
- `mode_mapping`: Dictionary mapping transport mode codes to names
- `emission_factors`: Environmental impact factors by transport mode
- `safety_factors`: Safety risk factors by transport mode
- `business_questions`: List of key business questions framework

#### Methods

##### extract_zip_files()
```python
def extract_zip_files(self) -> None
```
Extracts all ZIP files from yearly data directories.

**Behavior:**
- Processes years 2020-2024
- Creates `extracted/` subdirectories
- Organizes extracted files by ZIP filename
- Logs extraction progress and errors

**Dependencies:** zipfile, logging

##### process_csv_files(year)
```python
def process_csv_files(self, year: int) -> pd.DataFrame
```
Processes all CSV files for a specified year.

**Parameters:**
- `year` (int): Target year for processing (2020-2024)

**Returns:**
- `pd.DataFrame`: Combined dataset with standardized columns

**Processing Steps:**
1. Locates extracted CSV files
2. Handles date parsing from filenames/data
3. Converts VALUE and SHIPWT to numeric
4. Adds derived columns (Date, year, month)
5. Combines multiple files into single DataFrame

**Key Columns Processed:**
- `DISAGMOT`: Transport mode identifier
- `VALUE`: Freight monetary value (USD)
- `SHIPWT`: Shipment weight (tons)
- `USASTATE`, `MEXSTATE`, `CANPROV`: Geographic identifiers
- `MONTH`, `YEAR`: Temporal identifiers

##### analyze_data(df)
```python
def analyze_data(self, df: pd.DataFrame) -> Dict[str, Any]
```
Performs comprehensive analysis on freight data.

**Parameters:**
- `df` (pd.DataFrame): Preprocessed freight data

**Returns:**
- `Dict[str, Any]`: Analysis results with structured metrics

**Analysis Components:**

1. **Mode Analysis**
   ```python
   analysis['mode_analysis'] = {
       'metrics': {
           'MODE_NAME': {
               'value_sum': float,
               'value_mean': float,
               'shipment_count': int,
               'weight_sum': float,
               'weight_mean': float
           }
       },
       'modal_split': {'MODE_NAME': percentage}
   }
   ```

2. **Efficiency Analysis**
   ```python
   analysis['efficiency'] = {
       'MODE_NAME': {
           'mean': float,    # Average value per weight
           'median': float   # Median value per weight
       }
   }
   ```

3. **Environmental Impact**
   ```python
   analysis['environmental_impact'] = {
       'MODE_NAME': {
           'total_emissions': float,
           'emissions_per_ton': float
       }
   }
   ```

4. **Seasonal Patterns**
   ```python
   analysis['seasonal_patterns'] = {
       month: {
           'MODE_NAME': {
               'value': float,
               'weight': float
           }
       }
   }
   ```

5. **Trade Corridors**
   ```python
   analysis['trade_corridors'] = {
       'origin': {
           'MODE_NAME': {
               'value': float,
               'weight': float
           }
       }
   }
   ```

##### generate_insights(all_years_analysis)
```python
def generate_insights(self, all_years_analysis: Dict[int, Dict[str, Any]]) -> Dict[str, List[str]]
```
Generates business insights and explicit answers to key questions.

**Parameters:**
- `all_years_analysis` (Dict): Analysis results for all years

**Returns:**
- `Dict[str, List[str]]`: Structured insights and recommendations

**Insight Categories:**
- `trends`: Modal distribution trends
- `efficiency`: Cost-efficiency insights
- `environmental`: Environmental impact findings
- `seasonal`: Seasonal pattern insights
- `corridors`: Trade corridor analysis
- `recommendations`: Strategic recommendations
- `business_question_answers`: Explicit answers to 7 key questions

##### process_all_years()
```python
def process_all_years(self) -> Dict[str, Any]
```
Main orchestration method for complete analysis pipeline.

**Returns:**
- `Dict[str, Any]`: Complete analysis results or None on error

**Pipeline:**
1. Extract all ZIP files
2. Process each year (2020-2024)
3. Generate cross-year insights
4. Save results to JSON
5. Return structured results

## Data Schema

### Input Data Format
CSV files with the following required columns:

| Column | Type | Description |
|--------|------|-------------|
| DISAGMOT | int | Transport mode identifier (1-9) |
| VALUE | float | Freight value in USD |
| SHIPWT | float | Shipment weight in tons |
| USASTATE | str | US state code (optional) |
| MEXSTATE | str | Mexico state code (optional) |
| CANPROV | str | Canada province code (optional) |
| MONTH | int | Month (1-12, optional) |
| YEAR | int | Year (optional) |

### Transport Mode Mapping
```python
mode_mapping = {
    1: 'RAIL',
    3: 'ROAD', 
    4: 'AIR',
    5: 'VESSEL',
    6: 'MAIL',
    7: 'PIPELINE',
    8: 'OTHER',
    9: 'UNKNOWN'
}
```

### Emission Factors (kg CO2 per ton-mile)
```python
emission_factors = {
    'RAIL': 0.023,       # EPA estimates
    'ROAD': 0.161,       # EPA estimates  
    'AIR': 1.527,        # ICAO estimates
    'VESSEL': 0.048,     # IMO estimates
    'PIPELINE': 0.015,   # Industry average
    'MAIL': 0.161,       # Assume similar to road
    'OTHER': 0.161,      # Conservative estimate
    'UNKNOWN': 0.161     # Conservative estimate
}
```

## Configuration Management

### Current Configuration Issues
The current implementation has hardcoded configurations that should be externalized:

```python
# ISSUE: Hardcoded path
self.base_dir = Path(r"c:\Users\hbempong\TransBorderFreight_Analysis\data")
```

### Recommended Configuration Approach
```python
import os
from pathlib import Path

class FreightDataProcessor:
    def __init__(self, config_path=None):
        # Use environment variables or config file
        self.base_dir = Path(os.getenv('FREIGHT_DATA_DIR', './data'))
        self.output_dir = Path(os.getenv('FREIGHT_OUTPUT_DIR', './output'))
        
        # Load configuration from file if provided
        if config_path:
            self.load_config(config_path)
```

## Error Handling

### Current Error Handling Patterns
```python
try:
    # Operation
    logger.info("Success message")
except Exception as e:
    logger.error(f"Error message: {str(e)}")
    # Continue or return appropriate value
```

### Exception Types
- **File Processing Errors**: Missing files, corrupted data
- **Data Validation Errors**: Invalid data types, missing columns
- **Analysis Errors**: Calculation failures, empty datasets

## Performance Characteristics

### Memory Usage
- **Typical Dataset Size**: 100-500 MB per year
- **Peak Memory Usage**: ~2-3x dataset size during processing
- **Optimization**: Uses pandas chunking for large files

### Processing Time
- **Single Year**: 30-60 seconds
- **All Years (2020-2024)**: 3-5 minutes
- **Bottlenecks**: File I/O, data type conversions

### Scalability Considerations
- **CPU**: Single-threaded, could benefit from parallelization
- **Memory**: Loads entire year into memory (could use chunking)
- **Storage**: JSON output scales linearly with data size

## Testing Recommendations

### Unit Tests Needed
```python
def test_transport_mode_mapping():
    processor = FreightDataProcessor()
    assert processor.mode_mapping[1] == 'RAIL'
    assert processor.mode_mapping[5] == 'VESSEL'

def test_data_processing():
    # Test CSV processing with sample data
    pass

def test_analysis_calculations():
    # Test analysis logic with known inputs
    pass

def test_insight_generation():
    # Test insight generation with mock analysis
    pass
```

### Integration Tests
- End-to-end pipeline with sample datasets
- Error handling with corrupted/missing data
- Output validation and schema compliance

## Development Guidelines

### Code Style
- Follow PEP 8 conventions
- Use type hints for function signatures
- Maintain comprehensive logging
- Document complex business logic

### Adding New Analysis Features
1. **Extend analyze_data() method**
   ```python
   # Add new analysis section
   analysis['new_feature'] = calculate_new_metrics(df)
   ```

2. **Update insight generation**
   ```python
   # Add corresponding insights
   insights['new_category'].append("New insight")
   ```

3. **Update documentation**
   - Add to business questions if applicable
   - Document new metrics and calculations

### Adding New Transport Modes
1. Update `mode_mapping` dictionary
2. Add emission and safety factors
3. Update documentation and tests

## Deployment Considerations

### Environment Requirements
- Python 3.8+
- Sufficient disk space for data storage
- Memory: 4GB+ recommended for large datasets

### Configuration Management
- Use environment variables for paths
- Externalize emission factors and mappings
- Implement configuration validation

### Monitoring and Logging
- Implement structured logging
- Add performance metrics
- Monitor data quality indicators

## Future Enhancements

### Near-term Improvements
1. **Configuration Management**: External config files
2. **Test Suite**: Comprehensive unit and integration tests
3. **Documentation**: Enhanced inline documentation
4. **Error Handling**: More specific exception handling

### Long-term Enhancements
1. **Performance**: Parallel processing, data chunking
2. **Features**: Predictive analytics, anomaly detection
3. **Interface**: REST API, web dashboard
4. **Data**: Real-time processing capabilities