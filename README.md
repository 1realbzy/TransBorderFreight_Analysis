# TransBorder Freight Analysis (2020-2024)

## Project Overview
This project analyzes cross-border freight transportation data from 2020 to 2024, providing data-driven insights into transportation patterns, efficiency, and sustainability. The analysis focuses on answering seven key business questions with actionable recommendations.

## Key Features
- Interactive dashboards showing freight movement patterns
- Data-driven recommendations for optimization
- Year-over-year trend analysis
- Modal comparison and efficiency metrics

## Key Business Questions & Answers

### Q1: Major Trends in Freight Value and Volume
- Vessel transport dominates with 65.0% of total freight value
- Mail services handle 12.9% of value
- Rail contributes 7.5% of total value
- Heavy reliance on maritime transport for international freight

### Q2: Cost-Efficiency and Environmental Sustainability
- Most cost-efficient: Road ($1,123.06/ton) and Rail ($83.06/ton)
- Most environmentally sustainable: Pipeline (0.01 kg CO2/ton) and Rail (0.02 kg CO2/ton)
- Rail offers the best balance of cost efficiency and environmental sustainability

### Q3: Seasonal Patterns and Implications
- Peak activity occurs in May
- Suggests potential capacity constraints during peak periods
- Infrastructure planning should account for seasonal variations

### Q4: High-Growth Trade Corridors
Top trade corridors by value:
- Texas: $911.3B
- Michigan: $440.5B
- California: $315.2B
These corridors should be prioritized for infrastructure investment

### Q5: Safety Risks and Environmental Impacts
Environmental Impact Rankings (kg CO2 per ton):
- Air: 1.53
- Road/Mail: 0.16
- Rail: 0.02
- Pipeline: 0.01

### Q6: Modal Split Evolution and Infrastructure Implications
Current modal split suggests:
- Heavy reliance on maritime infrastructure
- Need for better intermodal connections
- Potential capacity constraints in major ports

### Q7: Strategic Recommendations
1. Invest in port infrastructure to support dominant vessel traffic
2. Develop intermodal connections for efficient transfers
3. Implement emission reduction strategies for high-impact modes
4. Optimize capacity utilization during peak seasons
5. Focus on rail infrastructure for sustainable growth

## Detailed Analysis by Category

### Transport Mode Trends
- Vessel: 65.0% of total freight value
- Mail: 12.9%
- Rail: 7.5%
- Pipeline: 6.8%
- Other: 3.7%
- Road: 3.4%
- Unknown: 0.7%
- Air: <0.1%

### Efficiency Metrics (Value per Ton)
- Other: $2,905.32
- Road: $1,123.06
- Unknown: $123.58
- Rail: $83.06
- Air: $78.66
- Vessel: $36.90
- Mail: $2.06
- Pipeline: $0.20

### Environmental Impact (kg CO2 per ton)
- Air: 1.53
- Road/Mail/Other: 0.16
- Vessel: 0.05
- Rail: 0.02
- Pipeline: 0.01

### Major Trade Corridors
1. Texas: $911.3B
2. Michigan: $440.5B
3. California: $315.2B
4. Illinois: $288.1B
5. Ohio: $142.6B
6. DU: $136.4B
7. New York: $113.2B

## Strategic Recommendations

### Infrastructure Development
- Expand port capacity in major maritime hubs
- Develop intermodal connections for efficient transfers
- Invest in rail infrastructure for sustainable growth

### Environmental Sustainability
- Implement emission reduction strategies for air and road transport
- Promote modal shift to rail and pipeline where feasible
- Develop green corridors for high-volume routes

### Operational Efficiency
- Optimize scheduling to better distribute seasonal freight volumes
- Focus safety measures on modes with higher risk factors
- Develop specialized handling for high-value corridors

### Data and Monitoring
- Enhance data collection systems
- Implement real-time monitoring of key corridors
- Develop predictive analytics for capacity planning

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
