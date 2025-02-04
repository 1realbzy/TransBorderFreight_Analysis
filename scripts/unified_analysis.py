"""
Unified Analysis Script for TransBorder Freight Data
Handles all objectives efficiently in a single pass through the data
"""
import pandas as pd
import numpy as np
from pathlib import Path
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
from typing import Dict, List, Tuple
from config import *
from detailed_patterns_analysis import DetailedPatternsAnalysis
import logging
from data_fetcher import TransBorderDataFetcher
from prepare_data import combine_freight_data, validate_data_coverage
from crisp_dm_analysis import CRISPDMAnalysis

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class UnifiedFreightAnalysis:
    def __init__(self, data_dir: Path):
        self.data_dir = data_dir
        self.output_dir = data_dir.parent / "output" / "unified_analysis"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.patterns_analysis = DetailedPatternsAnalysis(data_dir)
        self.results_cache = {}

    def analyze_all_objectives(self, years: List[str] = ANALYSIS_YEARS):
        """Run complete analysis for all objectives efficiently"""
        print(f"Starting unified analysis for years: {years}")
        
        # Initialize results storage
        self.results = {
            'patterns': {},
            'efficiency': {},
            'environmental': {},
            'safety': {},
            'economic': {}
        }
        
        for year in years:
            print(f"\nAnalyzing year {year}...")
            
            # Get base data efficiently (single pass)
            base_data = self._get_base_data(year)
            
            # Analyze all aspects in parallel
            self._analyze_patterns(base_data, year)
            self._analyze_efficiency(base_data, year)
            self._analyze_environmental(base_data, year)
            self._analyze_safety(base_data, year)
            self._analyze_economic(base_data, year)
            
            # Clear memory after each year
            del base_data
        
        # Generate cross-year insights
        self._generate_trend_analysis(years)
        
    def _get_base_data(self, year: str) -> pd.DataFrame:
        """Efficiently load and preprocess data for a given year"""
        files = list(self.data_dir.glob(f"{year}/**/*.csv"))
        dfs = []
        
        for file in files:
            try:
                # Try different encodings
                for encoding in ['utf-8', 'latin1', 'iso-8859-1', 'cp1252']:
                    try:
                        df = pd.read_csv(file, dtype=str, encoding=encoding)
                        break
                    except UnicodeDecodeError:
                        continue
                    except Exception as e:
                        print(f"Error reading {file} with {encoding}: {str(e)}")
                        continue
                
                # Convert numeric columns
                numeric_cols = ['VALUE', 'SHIPWT', 'FREIGHT_CHARGES']
                for col in numeric_cols:
                    if col in df.columns:
                        df[col] = pd.to_numeric(df[col].str.replace(',', ''), errors='coerce').fillna(0)
                
                # Add metadata
                df['source_file'] = file.name
                df['year'] = year
                df['month'] = file.name[-6:-4]
                
                # Add origin/destination for trade corridor analysis
                if 'USASTATE' in df.columns:
                    df['origin'] = df['USASTATE']
                    df['destination'] = df.apply(
                        lambda x: x.get('MEXSTATE', '') or x.get('CANPROV', '') or x.get('COUNTRY', ''),
                        axis=1
                    )
                
                dfs.append(df)
            except Exception as e:
                print(f"Error processing {file}: {str(e)}")
                continue
        
        if not dfs:
            print(f"Warning: No data loaded for year {year}")
            return pd.DataFrame()
            
        return pd.concat(dfs, ignore_index=True)

    def _analyze_patterns(self, df: pd.DataFrame, year: str):
        """Analyze trade patterns (Objective 1)"""
        if df.empty:
            self.results['patterns'][year] = {
                'seasonal': {'total_value': 0, 'total_shipments': 0, 'peak_month': None, 'low_month': None},
                'corridors': {'top_corridors': [], 'total_corridors': 0},
                'commodities': {'top_commodities': [], 'total_commodities': 0}
            }
            return
            
        self.results['patterns'][year] = {
            'seasonal': self._analyze_seasonal_patterns(df),
            'corridors': self._analyze_trade_corridors(df),
            'commodities': self._analyze_commodity_distribution(df)
        }

    def _analyze_efficiency(self, df: pd.DataFrame, year: str):
        """Analyze operational efficiency (Objective 2)"""
        if df.empty:
            self.results['efficiency'][year] = {
                'metrics': {
                    'value_density': 0,
                    'avg_shipment_value': 0,
                    'total_freight_charges': 0,
                    'shipments_count': 0
                },
                'high_cost_routes': []
            }
            return
            
        # Calculate efficiency metrics
        efficiency_metrics = {
            'value_density': df['VALUE'].sum() / df['SHIPWT'].sum() if df['SHIPWT'].sum() > 0 else 0,
            'avg_shipment_value': df['VALUE'].mean(),
            'total_freight_charges': df['FREIGHT_CHARGES'].sum(),
            'shipments_count': len(df)
        }
        
        # Identify potential bottlenecks
        high_cost_routes = df.groupby(['origin', 'destination']).agg({
            'FREIGHT_CHARGES': 'mean',
            'VALUE': 'sum',
            'SHIPWT': 'sum'
        }).reset_index()
        
        self.results['efficiency'][year] = {
            'metrics': efficiency_metrics,
            'high_cost_routes': high_cost_routes.to_dict('records')
        }

    def _analyze_environmental(self, df: pd.DataFrame, year: str):
        """Analyze environmental impact (Objective 3)"""
        if df.empty:
            self.results['environmental'][year] = {
                'emissions': pd.DataFrame(),
                'total_emissions': 0
            }
            return
            
        # Calculate emissions by transport mode
        emissions = df.groupby('DISAGMOT').agg({
            'SHIPWT': 'sum',
            'VALUE': 'sum'
        }).reset_index()
        
        # Apply emissions factors
        emissions['estimated_emissions'] = emissions.apply(
            lambda x: x['SHIPWT'] * EMISSIONS_THRESHOLDS.get(
                TRANSPORT_MODES.get(x['DISAGMOT'], 'truck').lower(), 
                EMISSIONS_THRESHOLDS['truck']
            ),
            axis=1
        )
        
        self.results['environmental'][year] = {
            'emissions': emissions.to_dict('records'),
            'total_emissions': emissions['estimated_emissions'].sum()
        }

    def _analyze_safety(self, df: pd.DataFrame, year: str):
        """Analyze safety aspects (Objective 4)"""
        if df.empty:
            self.results['safety'][year] = {
                'high_risk_count': 0,
                'high_risk_value': 0,
                'hazmat_shipments': 0
            }
            return
            
        # Identify high-risk shipments
        high_risk = df[
            (df['VALUE'] > SAFETY_THRESHOLDS['high_risk_value']) |
            (df['SHIPWT'] > SAFETY_THRESHOLDS['weight_limit']) |
            (df['DEPE'].isin(SAFETY_THRESHOLDS['hazmat_codes']))
        ]
        
        self.results['safety'][year] = {
            'high_risk_count': len(high_risk),
            'high_risk_value': high_risk['VALUE'].sum(),
            'hazmat_shipments': len(df[df['DEPE'].isin(SAFETY_THRESHOLDS['hazmat_codes'])])
        }

    def _analyze_economic(self, df: pd.DataFrame, year: str):
        """Analyze economic impacts (Objective 5)"""
        if df.empty:
            self.results['economic'][year] = {
                'indicators': {
                    'total_trade_value': 0,
                    'monthly_variation': 0,
                    'trade_balance': 0
                },
                'monthly_trends': []
            }
            return
            
        monthly_totals = df.groupby('month').agg({
            'VALUE': 'sum',
            'SHIPWT': 'sum'
        }).reset_index()
        
        # Calculate economic indicators
        economic_indicators = {
            'total_trade_value': df['VALUE'].sum(),
            'monthly_variation': monthly_totals['VALUE'].std() / monthly_totals['VALUE'].mean() if len(monthly_totals) > 1 else 0,
            'trade_balance': (
                df[df['origin'].fillna('').str.startswith('US')]['VALUE'].sum() -
                df[df['destination'].fillna('').str.startswith('US')]['VALUE'].sum()
            )
        }
        
        self.results['economic'][year] = {
            'indicators': economic_indicators,
            'monthly_trends': monthly_totals.to_dict('records')
        }

    def _generate_trend_analysis(self, years: List[str]):
        """Generate cross-year trend analysis"""
        # Compile trends across all years
        self.trends = {
            'trade_growth': self._calculate_trade_growth(years),
            'efficiency_trends': self._calculate_efficiency_trends(years),
            'environmental_impact': self._calculate_environmental_trends(years),
            'safety_metrics': self._calculate_safety_trends(years),
            'economic_indicators': self._calculate_economic_trends(years)
        }
        
        # Save results
        self._save_results()

    def _calculate_trade_growth(self, years: List[str]) -> Dict:
        """Calculate trade growth trends across years"""
        growth_trends = {}
        for metric in ['total_value', 'shipment_count', 'value_density']:
            yearly_values = []
            for year in years:
                if metric == 'total_value':
                    value = self.results['patterns'][year]['seasonal'].get('total_value', 0)
                elif metric == 'shipment_count':
                    value = self.results['patterns'][year]['seasonal'].get('total_shipments', 0)
                else:  # value_density
                    if year in self.results['efficiency']:
                        value = self.results['efficiency'][year]['metrics'].get('value_density', 0)
                    else:
                        value = 0
                yearly_values.append(value)
            
            # Calculate growth rates
            growth_rates = []
            for i in range(1, len(yearly_values)):
                prev_value = yearly_values[i-1]
                curr_value = yearly_values[i]
                if prev_value > 0:
                    growth_rate = ((curr_value - prev_value) / prev_value) * 100
                else:
                    growth_rate = 0 if curr_value == 0 else 100  # Consider it 100% growth if starting from 0
                growth_rates.append(growth_rate)
            
            growth_trends[metric] = {
                'values': yearly_values,
                'growth_rates': growth_rates
            }
        return growth_trends

    def _calculate_efficiency_trends(self, years: List[str]) -> Dict:
        """Calculate efficiency trends across years"""
        return {
            year: self.results['efficiency'][year]['metrics']
            for year in years
        }

    def _calculate_environmental_trends(self, years: List[str]) -> Dict:
        """Calculate environmental impact trends"""
        return {
            year: {
                'total_emissions': self.results['environmental'][year]['total_emissions'],
                'by_mode': self.results['environmental'][year]['emissions']
            }
            for year in years
        }

    def _calculate_safety_trends(self, years: List[str]) -> Dict:
        """Calculate safety metric trends"""
        return {
            year: self.results['safety'][year]
            for year in years
        }

    def _calculate_economic_trends(self, years: List[str]) -> Dict:
        """Calculate economic indicator trends"""
        return {
            year: self.results['economic'][year]['indicators']
            for year in years
        }

    def _save_results(self):
        """Save analysis results and visualizations"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_file = self.output_dir / f"unified_analysis_results_{timestamp}.html"
        
        # Create comprehensive HTML report
        self._generate_html_report(results_file)
        print(f"\nAnalysis complete. Results saved to {results_file}")

    def _generate_html_report(self, output_file: Path):
        """Generate comprehensive HTML report with all analyses"""
        import plotly.io as pio
        
        # Create report sections
        sections = []
        
        # 1. Overview
        sections.append(self._create_overview_section())
        
        # 2. Trade Patterns
        sections.append(self._create_patterns_section())
        
        # 3. Efficiency Analysis
        sections.append(self._create_efficiency_section())
        
        # 4. Environmental Impact
        sections.append(self._create_environmental_section())
        
        # 5. Safety Analysis
        sections.append(self._create_safety_section())
        
        # 6. Economic Impact
        sections.append(self._create_economic_section())
        
        # 7. Recommendations
        sections.append(self._create_recommendations_section())
        
        # Combine all sections
        html_content = "\n".join(sections)
        
        # Save to file
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(html_content)

    def _create_overview_section(self) -> str:
        """Create overview section of the report"""
        return f"""
        <h1>TransBorder Freight Analysis Report</h1>
        <p>Analysis period: {min(ANALYSIS_YEARS)} - {max(ANALYSIS_YEARS)}</p>
        <h2>Executive Summary</h2>
        <ul>
            <li>Total Trade Value: ${self.trends['trade_growth']['total_value']['values'][-1]:,.2f}</li>
            <li>Average Annual Growth: {np.mean(self.trends['trade_growth']['total_value']['growth_rates']):,.1f}%</li>
            <li>Environmental Impact: {self.trends['environmental_impact'][max(ANALYSIS_YEARS)]['total_emissions']:,.0f} CO2e</li>
        </ul>
        """

    def _create_patterns_section(self) -> str:
        """Create trade patterns section of the report"""
        return f"""
        <h2>Trade Patterns Analysis</h2>
        <h3>Seasonal Trends</h3>
        <p>Analysis of monthly and quarterly patterns across years.</p>
        <h3>Trade Corridors</h3>
        <p>Key trading routes and their characteristics.</p>
        <h3>Commodity Distribution</h3>
        <p>Analysis of major commodity groups and their trade patterns.</p>
        """

    def _create_efficiency_section(self) -> str:
        """Create efficiency analysis section of the report"""
        return f"""
        <h2>Operational Efficiency Analysis</h2>
        <h3>Key Metrics</h3>
        <ul>
            <li>Average Value Density: ${self.trends['efficiency_trends'][max(ANALYSIS_YEARS)]['value_density']:,.2f}/kg</li>
            <li>Average Shipment Value: ${self.trends['efficiency_trends'][max(ANALYSIS_YEARS)]['avg_shipment_value']:,.2f}</li>
        </ul>
        <h3>Identified Inefficiencies</h3>
        <p>Analysis of bottlenecks and areas for improvement.</p>
        """

    def _create_environmental_section(self) -> str:
        """Create environmental impact section of the report"""
        latest_year = max(ANALYSIS_YEARS)
        emissions = self.trends['environmental_impact'][latest_year]
        return f"""
        <h2>Environmental Impact Analysis</h2>
        <h3>Emissions by Transport Mode</h3>
        <p>Total emissions: {emissions['total_emissions']:,.0f} CO2e</p>
        <h3>Sustainability Metrics</h3>
        <p>Analysis of environmental performance indicators.</p>
        """

    def _create_safety_section(self) -> str:
        """Create safety analysis section of the report"""
        latest_year = max(ANALYSIS_YEARS)
        safety = self.trends['safety_metrics'][latest_year]
        return f"""
        <h2>Safety Analysis</h2>
        <h3>Risk Assessment</h3>
        <ul>
            <li>High-Risk Shipments: {safety['high_risk_count']:,}</li>
            <li>Hazmat Shipments: {safety['hazmat_shipments']:,}</li>
        </ul>
        <h3>Safety Recommendations</h3>
        <p>Key recommendations for improving safety protocols.</p>
        """

    def _create_economic_section(self) -> str:
        """Create economic impact section of the report"""
        latest_year = max(ANALYSIS_YEARS)
        economic = self.trends['economic_indicators'][latest_year]
        return f"""
        <h2>Economic Impact Analysis</h2>
        <h3>Key Economic Indicators</h3>
        <ul>
            <li>Total Trade Value: ${economic['total_trade_value']:,.2f}</li>
            <li>Trade Balance: ${economic['trade_balance']:,.2f}</li>
        </ul>
        <h3>Economic Disruption Analysis</h3>
        <p>Analysis of major economic events and their impact.</p>
        """

    def _create_recommendations_section(self) -> str:
        """Create recommendations section of the report"""
        return """
        <h2>Recommendations</h2>
        <h3>Operational Improvements</h3>
        <ul>
            <li>Optimize high-cost trade corridors</li>
            <li>Implement more efficient routing strategies</li>
            <li>Enhance infrastructure at key bottleneck points</li>
        </ul>
        <h3>Environmental Sustainability</h3>
        <ul>
            <li>Increase use of low-emission transport modes</li>
            <li>Implement green logistics practices</li>
            <li>Optimize load factors to reduce empty runs</li>
        </ul>
        <h3>Safety Enhancements</h3>
        <ul>
            <li>Strengthen hazmat handling protocols</li>
            <li>Improve risk assessment procedures</li>
            <li>Enhance safety training programs</li>
        </ul>
        <h3>Economic Resilience</h3>
        <ul>
            <li>Diversify trade corridors</li>
            <li>Develop contingency routing plans</li>
            <li>Strengthen cross-border partnerships</li>
        </ul>
        """

    def _analyze_seasonal_patterns(self, df: pd.DataFrame) -> Dict:
        """Analyze seasonal patterns in the data"""
        monthly_stats = df.groupby('month').agg({
            'VALUE': ['sum', 'count', 'mean'],
            'SHIPWT': ['sum', 'mean']
        }).reset_index()
        
        # Flatten column names
        monthly_stats.columns = ['month', 'total_value', 'num_shipments', 'avg_value', 'total_weight', 'avg_weight']
        
        # Add month names
        month_names = {
            '01': 'January', '02': 'February', '03': 'March',
            '04': 'April', '05': 'May', '06': 'June',
            '07': 'July', '08': 'August', '09': 'September',
            '10': 'October', '11': 'November', '12': 'December'
        }
        monthly_stats['month_name'] = monthly_stats['month'].map(month_names)
        
        return {
            'monthly_stats': monthly_stats.to_dict('records'),
            'total_value': monthly_stats['total_value'].sum(),
            'total_shipments': monthly_stats['num_shipments'].sum(),
            'peak_month': monthly_stats.loc[monthly_stats['total_value'].idxmax(), 'month_name'],
            'low_month': monthly_stats.loc[monthly_stats['total_value'].idxmin(), 'month_name']
        }

    def _analyze_trade_corridors(self, df: pd.DataFrame) -> Dict:
        """Analyze trade corridors in the data"""
        corridor_stats = df.groupby(['origin', 'destination']).agg({
            'VALUE': ['sum', 'count'],
            'SHIPWT': 'sum'
        }).reset_index()
        
        # Flatten column names
        corridor_stats.columns = ['origin', 'destination', 'total_value', 'num_shipments', 'total_weight']
        
        return {
            'corridor_stats': corridor_stats.to_dict('records'),
            'top_corridors': corridor_stats.nlargest(5, 'total_value').to_dict('records'),
            'total_corridors': len(corridor_stats)
        }

    def _analyze_commodity_distribution(self, df: pd.DataFrame) -> Dict:
        """Analyze commodity distribution in the data"""
        commodity_stats = df.groupby('DEPE').agg({
            'VALUE': ['sum', 'count'],
            'SHIPWT': 'sum'
        }).reset_index()
        
        # Flatten column names
        commodity_stats.columns = ['commodity_code', 'total_value', 'num_shipments', 'total_weight']
        
        # Calculate percentages
        total_value = commodity_stats['total_value'].sum()
        total_weight = commodity_stats['total_weight'].sum()
        commodity_stats['value_percentage'] = (commodity_stats['total_value'] / total_value) * 100
        commodity_stats['weight_percentage'] = (commodity_stats['total_weight'] / total_weight) * 100
        
        return {
            'commodity_stats': commodity_stats.to_dict('records'),
            'top_commodities': commodity_stats.nlargest(5, 'total_value').to_dict('records'),
            'total_commodities': len(commodity_stats)
        }

def run_unified_analysis(base_dir: Path, fetch_missing: bool = True):
    """Run the complete analysis pipeline"""
    try:
        data_dir = base_dir / "data"
        output_dir = base_dir / "output"
        output_dir.mkdir(exist_ok=True)
        
        # Step 1: Check data coverage
        logger.info("Checking data coverage...")
        coverage = validate_data_coverage(data_dir)
        
        # Step 2: Fetch missing data if requested
        if fetch_missing:
            logger.info("Fetching missing data...")
            fetcher = TransBorderDataFetcher(base_dir)
            fetcher.fetch_missing_data()
            
            # Recheck coverage after fetching
            coverage = validate_data_coverage(data_dir)
        
        # Step 3: Prepare data
        logger.info("Preparing data...")
        combined_data = combine_freight_data(data_dir, output_dir)
        
        # Step 4: Run CRISP-DM analysis
        logger.info("Running CRISP-DM analysis...")
        analyzer = CRISPDMAnalysis(data_dir)
        results = analyzer.analyze_with_coverage_warning(combined_data)
        
        # Step 5: Save results
        results_file = output_dir / "analysis_results.json"
        pd.DataFrame(results).to_json(results_file)
        logger.info(f"Analysis results saved to {results_file}")
        
        return results
        
    except Exception as e:
        logger.error(f"Error in unified analysis: {str(e)}")
        raise

if __name__ == "__main__":
    base_dir = Path(__file__).parent.parent
    
    logger.info("Starting unified analysis...")
    results = run_unified_analysis(base_dir)
    logger.info("Analysis completed successfully")
