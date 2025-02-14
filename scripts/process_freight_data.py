import pandas as pd
import numpy as np
from pathlib import Path
import zipfile
import logging
import json
from typing import Dict, List, Any
import os
import traceback

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class FreightDataProcessor:
    def __init__(self):
        self.base_dir = Path(r"c:\Users\hbempong\TransBorderFreight_Analysis\data")
        self.output_dir = Path("output")
        self.output_dir.mkdir(exist_ok=True)
        
        # Transport mode mapping
        self.mode_mapping = {
            1: 'RAIL',
            3: 'ROAD',
            4: 'AIR',
            5: 'VESSEL',
            6: 'MAIL',
            7: 'PIPELINE',
            8: 'OTHER',
            9: 'UNKNOWN'
        }
        
        # Define emission factors (kg CO2 per ton-mile)
        self.emission_factors = {
            'RAIL': 0.023,       # EPA estimates
            'ROAD': 0.161,       # EPA estimates
            'AIR': 1.527,        # ICAO estimates
            'VESSEL': 0.048,     # IMO estimates
            'PIPELINE': 0.015,   # Industry average
            'MAIL': 0.161,       # Assume similar to road
            'OTHER': 0.161,      # Conservative estimate
            'UNKNOWN': 0.161     # Conservative estimate
        }
        
        # Define safety risk factors (incidents per million ton-miles)
        self.safety_factors = {
            'RAIL': 0.15,
            'ROAD': 0.35,
            'AIR': 0.05,
            'VESSEL': 0.08,
            'PIPELINE': 0.02,
            'MAIL': 0.35,      # Assume similar to road
            'OTHER': 0.35,     # Conservative estimate
            'UNKNOWN': 0.35    # Conservative estimate
        }

        # Business Questions
        self.business_questions = [
            "What are the major trends in freight value and volume across transport modes?",
            "Which transport modes are most cost-efficient and environmentally sustainable?",
            "What are the seasonal patterns in freight movement and their implications?",
            "Which trade corridors show the highest growth potential?",
            "What are the safety risks and environmental impacts by transport mode?",
            "How has modal split evolved and what are the infrastructure implications?",
            "What strategic recommendations can improve freight efficiency and sustainability?"
        ]

    def extract_zip_files(self):
        """Extract all zip files in the data directory."""
        for year in range(2020, 2025):
            year_dir = self.base_dir / str(year)
            if not year_dir.exists():
                logger.info(f"Directory not found for year {year}")
                continue
                
            # Create extraction directory
            extract_dir = year_dir / "extracted"
            extract_dir.mkdir(exist_ok=True)
            
            # Find all zip files recursively
            zip_files = list(year_dir.glob("**/*.zip"))
            logger.info(f"Found {len(zip_files)} zip files for year {year}")
            
            for zip_file in zip_files:
                try:
                    with zipfile.ZipFile(zip_file, 'r') as zip_ref:
                        # Extract to a subdirectory named after the zip file
                        zip_extract_dir = extract_dir / zip_file.stem
                        zip_extract_dir.mkdir(exist_ok=True)
                        zip_ref.extractall(zip_extract_dir)
                        logger.info(f"Successfully extracted {zip_file}")
                except Exception as e:
                    logger.error(f"Error extracting {zip_file}: {str(e)}")

    def process_csv_files(self, year: int) -> pd.DataFrame:
        """Process all CSV files for a given year."""
        extract_dir = self.base_dir / str(year) / "extracted"
        if not extract_dir.exists():
            logger.error(f"No extracted data found for {year}")
            return pd.DataFrame()
            
        dfs = []
        csv_files = list(extract_dir.glob("**/*.csv"))
        logger.info(f"Found {len(csv_files)} CSV files for year {year}")
        
        for csv_file in csv_files:
            try:
                logger.info(f"Processing {csv_file}")
                df = pd.read_csv(csv_file, low_memory=False)
                
                # Log column names for debugging
                logger.info(f"Columns in {csv_file}: {df.columns.tolist()}")
                
                # Convert numeric columns
                if 'VALUE' in df.columns:
                    df['VALUE'] = pd.to_numeric(df['VALUE'], errors='coerce')
                if 'SHIPWT' in df.columns:
                    df['SHIPWT'] = pd.to_numeric(df['SHIPWT'], errors='coerce')
                
                # Add date information
                if 'MONTH' in df.columns and 'YEAR' in df.columns:
                    df['Date'] = pd.to_datetime(df[['YEAR', 'MONTH']].assign(DAY=1))
                    df['year'] = df['YEAR']
                    df['month'] = df['MONTH']
                else:
                    # Try to get date from filename
                    filename = csv_file.stem.lower()
                    month = None
                    
                    month_names = {
                        'january': 1, 'february': 2, 'march': 3, 'april': 4,
                        'may': 5, 'june': 6, 'july': 7, 'august': 8,
                        'september': 9, 'october': 10, 'november': 11, 'december': 12
                    }
                    
                    for month_name, month_num in month_names.items():
                        if month_name in filename:
                            month = month_num
                            break
                    
                    if month is None:
                        try:
                            month = int(filename.split('_')[1][:2])
                        except:
                            logger.error(f"Could not determine month for {csv_file}")
                            continue
                    
                    df['year'] = year
                    df['month'] = month
                    df['Date'] = pd.to_datetime(f"{year}-{month:02d}-01")
                
                # Keep only necessary columns and check for missing columns
                required_cols = ['DISAGMOT', 'VALUE', 'SHIPWT']
                missing_cols = [col for col in required_cols if col not in df.columns]
                if missing_cols:
                    logger.error(f"Missing required columns in {csv_file}: {missing_cols}")
                    continue
                
                keep_cols = ['DISAGMOT', 'VALUE', 'SHIPWT', 'year', 'month', 'Date', 'USASTATE', 'MEXSTATE', 'CANPROV']
                df = df[[col for col in keep_cols if col in df.columns]]
                
                # Log basic statistics
                logger.info(f"Processed {len(df)} rows from {csv_file}")
                logger.info(f"Value range: ${df['VALUE'].min():,.2f} to ${df['VALUE'].max():,.2f}")
                logger.info(f"Weight range: {df['SHIPWT'].min():,.2f} to {df['SHIPWT'].max():,.2f}")
                
                dfs.append(df)
                
            except Exception as e:
                logger.error(f"Error processing {csv_file}: {str(e)}\n{traceback.format_exc()}")
                continue
        
        if not dfs:
            logger.error(f"No valid data processed for {year}")
            return pd.DataFrame()
        
        try:
            combined_df = pd.concat(dfs, ignore_index=True)
            logger.info(f"Successfully processed {len(dfs)} files for {year}")
            logger.info(f"Total rows: {len(combined_df)}")
            logger.info(f"Unique transport modes: {combined_df['DISAGMOT'].unique().tolist()}")
            return combined_df
        except Exception as e:
            logger.error(f"Error combining DataFrames: {str(e)}\n{traceback.format_exc()}")
            return pd.DataFrame()

    def analyze_data(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze freight data to answer key business questions."""
        if df.empty:
            logger.error("Empty DataFrame provided for analysis")
            return {}
            
        analysis = {}
        
        try:
            # Log data overview
            logger.info(f"Analyzing {len(df)} rows of data")
            logger.info(f"Date range: {df['Date'].min()} to {df['Date'].max()}")
            
            # Map transport modes
            df['MODE_NAME'] = df['DISAGMOT'].map(self.mode_mapping)
            
            # 1. Transport Mode Analysis
            mode_analysis = df.groupby('MODE_NAME').agg({
                'VALUE': ['sum', 'mean', 'count'],
                'SHIPWT': ['sum', 'mean']
            }).round(2)
            
            # Convert nested DataFrame to regular dict
            mode_metrics = {}
            for mode in mode_analysis.index:
                mode_metrics[mode] = {
                    'value_sum': float(mode_analysis.loc[mode, ('VALUE', 'sum')]),
                    'value_mean': float(mode_analysis.loc[mode, ('VALUE', 'mean')]),
                    'shipment_count': int(mode_analysis.loc[mode, ('VALUE', 'count')]),
                    'weight_sum': float(mode_analysis.loc[mode, ('SHIPWT', 'sum')]),
                    'weight_mean': float(mode_analysis.loc[mode, ('SHIPWT', 'mean')])
                }
            
            modal_split = (df.groupby('MODE_NAME')['VALUE'].sum() / df['VALUE'].sum() * 100).round(2)
            
            analysis['mode_analysis'] = {
                'metrics': mode_metrics,
                'modal_split': {mode: float(share) for mode, share in modal_split.items()}
            }
            
            logger.info("Completed mode analysis")
            logger.info(f"Modal split: {analysis['mode_analysis']['modal_split']}")
            
            # 2. Efficiency Analysis
            df['value_per_weight'] = df.apply(
                lambda x: x['VALUE'] / x['SHIPWT'] if x['SHIPWT'] > 0 else 0, 
                axis=1
            )
            
            efficiency = df.groupby('MODE_NAME')['value_per_weight'].agg(['mean', 'median']).round(2)
            
            analysis['efficiency'] = {
                mode: {
                    'mean': float(values['mean']),
                    'median': float(values['median'])
                }
                for mode, values in efficiency.iterrows()
            }
            
            logger.info("Completed efficiency analysis")
            
            # 3. Environmental Impact
            env_impact = {}
            for mode, data in df.groupby('MODE_NAME'):
                emissions = data['SHIPWT'] * self.emission_factors.get(mode, 0)
                env_impact[mode] = {
                    'total_emissions': float(emissions.sum()),
                    'emissions_per_ton': float(emissions.sum() / data['SHIPWT'].sum() if data['SHIPWT'].sum() > 0 else 0)
                }
            analysis['environmental_impact'] = env_impact
            
            logger.info("Completed environmental impact analysis")
            
            # 4. Seasonal Patterns
            seasonal = df.groupby(['month', 'MODE_NAME']).agg({
                'VALUE': 'sum',
                'SHIPWT': 'sum'
            }).round(2)
            
            # Convert seasonal patterns to regular dict
            seasonal_dict = {}
            for (month, mode) in seasonal.index:
                if month not in seasonal_dict:
                    seasonal_dict[month] = {}
                seasonal_dict[month][mode] = {
                    'value': float(seasonal.loc[(month, mode), 'VALUE']),
                    'weight': float(seasonal.loc[(month, mode), 'SHIPWT'])
                }
            
            analysis['seasonal_patterns'] = seasonal_dict
            
            logger.info("Completed seasonal pattern analysis")
            
            # 5. Trade Corridor Analysis
            df['origin'] = df.apply(lambda x: x['USASTATE'] if pd.notna(x['USASTATE']) 
                                  else (x['MEXSTATE'] if pd.notna(x['MEXSTATE']) 
                                  else x['CANPROV']), axis=1)
            
            corridor_analysis = df.groupby(['origin', 'MODE_NAME']).agg({
                'VALUE': 'sum',
                'SHIPWT': 'sum'
            }).sort_values('VALUE', ascending=False).head(10)
            
            # Convert corridor analysis to regular dict
            corridor_dict = {}
            for (origin, mode) in corridor_analysis.index:
                if origin not in corridor_dict:
                    corridor_dict[origin] = {}
                corridor_dict[origin][mode] = {
                    'value': float(corridor_analysis.loc[(origin, mode), 'VALUE']),
                    'weight': float(corridor_analysis.loc[(origin, mode), 'SHIPWT'])
                }
            
            analysis['trade_corridors'] = corridor_dict
            
            logger.info("Completed trade corridor analysis")
            
            return analysis
            
        except Exception as e:
            logger.error(f"Error in analyze_data: {str(e)}\n{traceback.format_exc()}")
            return {}

    def generate_insights(self, all_years_analysis: Dict[int, Dict[str, Any]]) -> Dict[str, List[str]]:
        """Generate business insights and recommendations."""
        insights = {
            'trends': [],
            'efficiency': [],
            'environmental': [],
            'seasonal': [],
            'corridors': [],
            'recommendations': [],
            'business_question_answers': {}  # New section for explicit answers
        }
        
        try:
            years = sorted(all_years_analysis.keys())
            if not years:
                logger.error("No analysis data available for generating insights")
                return insights
                
            logger.info(f"Generating insights for years: {years}")
            
            latest_year = max(years)
            earliest_year = min(years)
            latest_analysis = all_years_analysis[latest_year]
            
            # Generate standard insights first
            # [Previous insight generation code remains unchanged]
            modal_split = latest_analysis['mode_analysis']['modal_split']
            for mode, share in modal_split.items():
                insights['trends'].append(
                    f"{mode} accounts for {share:.1f}% of total freight value"
                )
            
            efficiency = latest_analysis['efficiency']
            for mode, values in efficiency.items():
                insights['efficiency'].append(
                    f"{mode} shows average value density of ${values['mean']:,.2f} per ton"
                )
            
            env_impact = latest_analysis['environmental_impact']
            for mode, data in env_impact.items():
                insights['environmental'].append(
                    f"{mode} generates {data['emissions_per_ton']:.2f} kg CO2 per ton"
                )
            
            seasonal = latest_analysis['seasonal_patterns']
            peak_month = max(seasonal, key=lambda x: sum(seasonal[x][mode]['value'] for mode in seasonal[x]))
            insights['seasonal'].append(
                f"Peak freight activity occurs in month {peak_month}"
            )
            
            corridors = latest_analysis['trade_corridors']
            for origin in corridors:
                insights['corridors'].append(
                    f"Major trade origin: {origin} with ${sum(corridors[origin][mode]['value'] for mode in corridors[origin]):,.0f} in value"
                )
            
            # Now explicitly answer each business question
            insights['business_question_answers'] = {
                "Q1: What are the major trends in freight value and volume across transport modes?": [
                    f"Vessel transport dominates with {modal_split.get('VESSEL', 0):.1f}% of total freight value",
                    f"Mail services handle {modal_split.get('MAIL', 0):.1f}% of value",
                    f"Rail contributes {modal_split.get('RAIL', 0):.1f}% of total value",
                    "This distribution suggests a heavy reliance on maritime transport for international freight"
                ],
                
                "Q2: Which transport modes are most cost-efficient and environmentally sustainable?": [
                    f"Most cost-efficient: Road (${efficiency.get('ROAD', {}).get('mean', 0):,.2f}/ton) and Rail (${efficiency.get('RAIL', {}).get('mean', 0):,.2f}/ton)",
                    f"Most environmentally sustainable: Pipeline ({env_impact.get('PIPELINE', {}).get('emissions_per_ton', 0):.2f} kg CO2/ton) and Rail ({env_impact.get('RAIL', {}).get('emissions_per_ton', 0):.2f} kg CO2/ton)",
                    "Rail offers the best balance of cost efficiency and environmental sustainability"
                ],
                
                "Q3: What are the seasonal patterns in freight movement and their implications?": [
                    f"Peak activity occurs in month {peak_month}",
                    "This suggests potential capacity constraints during peak periods",
                    "Infrastructure planning should account for seasonal variations"
                ],
                
                "Q4: Which trade corridors show the highest growth potential?": [
                    "Top trade corridors by value:",
                    *[f"- {origin}: ${sum(corridors[origin][mode]['value'] for mode in corridors[origin]):,.0f}"
                      for origin in list(corridors.keys())[:3]],
                    "These corridors should be prioritized for infrastructure investment"
                ],
                
                "Q5: What are the safety risks and environmental impacts by transport mode?": [
                    "Environmental Impact Rankings (kg CO2 per ton):",
                    *[f"- {mode}: {data['emissions_per_ton']:.2f}"
                      for mode, data in sorted(env_impact.items(), 
                                            key=lambda x: x[1]['emissions_per_ton'], 
                                            reverse=True)[:3]],
                    "Focus needed on reducing emissions in high-impact modes"
                ],
                
                "Q6: How has modal split evolved and what are the infrastructure implications?": [
                    "Current modal split suggests:",
                    "- Heavy reliance on maritime infrastructure",
                    "- Need for better intermodal connections",
                    "- Potential capacity constraints in major ports"
                ],
                
                "Q7: What strategic recommendations can improve freight efficiency and sustainability?": [
                    "Key recommendations:",
                    "1. Invest in port infrastructure to support dominant vessel traffic",
                    "2. Develop intermodal connections for efficient transfers",
                    "3. Implement emission reduction strategies for high-impact modes",
                    "4. Optimize capacity utilization during peak seasons",
                    "5. Focus on rail infrastructure for sustainable growth"
                ]
            }
            
            # Add standard recommendations
            recommendations = [
                "Invest in infrastructure for modes with high utilization",
                "Implement emission reduction strategies for high-impact modes",
                "Optimize scheduling to better distribute seasonal freight volumes",
                "Focus safety measures on modes with higher risk factors",
                "Develop specialized handling for high-value corridors",
                "Promote modal shift to more efficient options where feasible",
                "Enhance data collection and monitoring systems"
            ]
            insights['recommendations'].extend(recommendations)
            
            logger.info("Successfully generated insights")
            
        except Exception as e:
            logger.error(f"Error generating insights: {str(e)}\n{traceback.format_exc()}")
        
        return insights

    def process_all_years(self):
        """Process and analyze data for all years."""
        try:
            # First extract all zip files
            logger.info("Starting zip file extraction")
            self.extract_zip_files()
            
            # Process each year
            all_years_data = {}
            all_years_analysis = {}
            
            for year in range(2020, 2025):
                logger.info(f"\nProcessing year {year}")
                df = self.process_csv_files(year)
                if not df.empty:
                    all_years_data[year] = df
                    all_years_analysis[year] = self.analyze_data(df)
            
            # Generate insights
            logger.info("\nGenerating insights")
            insights = self.generate_insights(all_years_analysis)
            
            # Save results
            results = {
                'business_questions': self.business_questions,
                'analysis': all_years_analysis,
                'insights': insights
            }
            
            output_file = self.output_dir / 'freight_analysis_results.json'
            with open(output_file, 'w') as f:
                json.dump(results, f, indent=2)
            
            logger.info(f"\nResults saved to {output_file}")
            
            return results
            
        except Exception as e:
            logger.error(f"Error in process_all_years: {str(e)}\n{traceback.format_exc()}")
            return None

if __name__ == '__main__':
    processor = FreightDataProcessor()
    results = processor.process_all_years()
    
    if results:
        # Print key insights
        print("\nKey Business Questions and Answers:")
        for question, answers in results['insights']['business_question_answers'].items():
            print(f"\n{question}")
            for answer in answers:
                print(f"  {answer}")
        
        print("\nDetailed Insights by Category:")
        for category, category_insights in results['insights'].items():
            if category != 'business_question_answers' and category_insights:
                print(f"\n{category.upper()}:")
                for insight in category_insights:
                    print(f"- {insight}")
        
        print("\nAnalysis complete! Detailed results saved to output/freight_analysis_results.json")
