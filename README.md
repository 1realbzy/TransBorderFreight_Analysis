# TransBorder Freight Analysis (2020-2024)

## Project Overview
This project analyzes cross-border freight transportation data from 2020 to 2024, focusing on:
- Operational efficiency and cost optimization
- Environmental sustainability and emissions reduction
- Safety and risk management
- Economic performance and trade patterns

## Key Features
- Interactive dashboards showing freight movement patterns
- Data-driven recommendations for optimization
- Year-over-year trend analysis
- Modal comparison and efficiency metrics

## Setup
1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Run the analysis:
```bash
python scripts/crisp_dm_analysis.py
```

3. View dashboards:
```bash
python scripts/visualize_analysis.py
```
Access dashboards at:
- Main Dashboard: http://127.0.0.1:8050
- Recommendations: http://127.0.0.1:8051

## Project Structure
- `scripts/`: Analysis and visualization code
- `data/`: Raw freight data (not included in repo)
- `output/`: Analysis results and processed data

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

- Business Questions
- 

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
