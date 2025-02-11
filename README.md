# TransBorder Freight Analysis Project

## Project Overview
This project analyzes transportation data from the Bureau of Transportation Statistics (BTS) to uncover insights related to efficiency, safety, and environmental impacts of freight transportation. Using the CRISP-DM methodology, we aim to identify inefficiencies, recognize patterns, and propose actionable solutions to improve the performance and sustainability of transportation systems.

## Project Structure
```
TransBorderFreight_Analysis/
├── data/               # Raw data files organized by year and month
├── output/            # Processed data and analysis results
│   ├── analysis_results/  # Detailed analysis for each year
│   └── freight_data_*_processed.parquet  # Processed data files
├── scripts/           # Analysis scripts
│   ├── prepare_data.py     # Data preprocessing
│   ├── crisp_dm_analysis.py  # Main analysis script
│   └── organize_data.py    # Data organization utilities
└── README.md          # Project documentation
```

## Key Objectives
1. **Freight Movement Patterns**: Analyze trends in volume, routing, and transportation modes
2. **Operational Efficiency**: Identify and address inefficiencies in transportation systems
3. **Environmental Impact**: Assess emissions and sustainability metrics
4. **Safety and Risk Assessment**: Analyze high-value shipments and risk patterns
5. **Economic Disruptions**: Study impact on freight movements and efficiency
6. **Data-Driven Recommendations**: Provide actionable insights for improvement

## Analysis Components

### 1. Data Preparation
- Organized raw data files by year and month.
- Cleaned and standardized data formats.
- Added derived metrics like value density and cost efficiency.
- Converted processed data to efficient parquet format.

### 2. Movement Pattern Analysis
- Analyzed transport mode distributions
- Identified top shipping routes
- Studied seasonal patterns in freight movement

### 3. Efficiency Analysis
- Calculated value density across transport modes
- Analyzed regional efficiency patterns
- Identified opportunities for cost optimization

### 4. Environmental Impact Assessment
- Estimated emissions by transport mode
- Analyzed environmental efficiency of different routes
- Identified opportunities for emissions reduction

### 5. Safety and Risk Analysis
- Identified high-value shipment patterns
- Analyzed risk distribution across modes and routes
- Developed risk assessment metrics

### 6. Economic Analysis
- Tracked value trends over time
- Analyzed trade balance patterns
- Studied regional economic impacts

## Tools and Dependencies
- Python 3.8+
- pandas: Data manipulation and analysis
- numpy: Numerical computations
- pyarrow: Parquet file handling
- logging: Execution logging

## Usage
1. Run `organize_data.py` to structure raw data files
2. Execute `prepare_data.py` to process and clean the data
3. Run `crisp_dm_analysis.py` to perform comprehensive analysis

## Results
Analysis results are stored in JSON format in the `output/analysis_results/` directory:
- Individual year analyses: `analysis_YYYY.json`
- Combined analysis: `all_years_analysis.json`

## Future Work
1. Enhance environmental impact modeling
2. Add predictive analytics capabilities
3. Develop interactive visualization dashboard
4. Integrate real-time data analysis
5. Expand risk assessment metrics

## Contributing
1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License
This project is licensed under the MIT License - see the LICENSE file for details.
