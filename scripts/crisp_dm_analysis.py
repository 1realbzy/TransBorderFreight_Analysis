import pandas as pd
import numpy as np
from pathlib import Path
import logging
from typing import Dict, Any, List, Optional
import json
from datetime import datetime

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class FreightAnalysis:
    def __init__(self, output_dir: Path):
        """Initialize analysis with output directory."""
        self.output_dir = Path(output_dir)
        self.results_dir = self.output_dir / 'analysis_results'
        self.results_dir.mkdir(exist_ok=True)

    def analyze_freight_movements(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze freight movement patterns (Objective 1)."""
        try:
            results = {}
            
            # Analyze transport modes
            mode_stats = df.groupby('DISAGMOT').agg({
                'VALUE': ['sum', 'mean'],
                'SHIPWT': ['sum', 'mean']
            }).round(2)
            
            # Convert multi-index columns to string keys
            mode_stats_dict = {}
            for col in mode_stats.columns:
                key = f"{col[0]}_{col[1]}"
                mode_stats_dict[key] = mode_stats[col].to_dict()
            results['transport_mode_analysis'] = mode_stats_dict
            
            # Analyze top routes
            df['route'] = df['USASTATE'] + '-' + df['MEXSTATE'].fillna('') + df['CANPROV'].fillna('')
            top_routes = df.groupby('route').agg({
                'VALUE': 'sum',
                'SHIPWT': 'sum'
            }).sort_values('VALUE', ascending=False).head(10)
            results['top_routes'] = top_routes.to_dict()
            
            # Analyze seasonal patterns
            df['month'] = pd.to_datetime(df['Date']).dt.month
            seasonal_patterns = df.groupby('month').agg({
                'VALUE': 'sum',
                'SHIPWT': 'sum'
            }).round(2)
            results['seasonal_patterns'] = seasonal_patterns.to_dict()
            
            return results
        except Exception as e:
            logger.error(f"Error analyzing freight movements: {str(e)}")
            return {}

    def analyze_operational_efficiency(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze operational inefficiencies (Objective 2)."""
        try:
            results = {}
            
            # Analyze value density across modes
            mode_efficiency = df.groupby('DISAGMOT').agg({
                'value_density': ['mean', 'std'],
                'cost_per_value': ['mean', 'std']
            }).round(4)
            
            # Convert multi-index columns to string keys
            mode_efficiency_dict = {}
            for col in mode_efficiency.columns:
                key = f"{col[0]}_{col[1]}"
                mode_efficiency_dict[key] = mode_efficiency[col].to_dict()
            results['mode_efficiency'] = mode_efficiency_dict
            
            # Analyze regional efficiency
            regional_efficiency = df.groupby(['USASTATE']).agg({
                'value_density': 'mean',
                'cost_per_value': 'mean'
            }).round(4)
            results['regional_efficiency'] = regional_efficiency.to_dict()
            
            return results
        except Exception as e:
            logger.error(f"Error analyzing operational efficiency: {str(e)}")
            return {}

    def analyze_environmental_impact(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze environmental impact (Objective 3)."""
        try:
            results = {}
            
            # Calculate emissions by transport mode (simplified model)
            # Using basic emission factors (for demonstration)
            emission_factors = {
                1: 2.5,  # Truck (kg CO2 per ton-km)
                2: 0.9,  # Rail
                3: 12.0, # Air
                4: 0.4   # Water
            }
            
            df['estimated_emissions'] = df.apply(
                lambda x: x['SHIPWT'] * emission_factors.get(x['DISAGMOT'], 0),
                axis=1
            )
            
            mode_emissions = df.groupby('DISAGMOT').agg({
                'estimated_emissions': ['sum', 'mean'],
                'SHIPWT': 'sum'
            }).round(2)
            
            # Convert multi-index columns to string keys
            mode_emissions_dict = {}
            for col in mode_emissions.columns:
                key = f"{col[0]}_{col[1]}" if isinstance(col, tuple) else col
                mode_emissions_dict[key] = mode_emissions[col].to_dict()
            results['emissions_by_mode'] = mode_emissions_dict
            
            return results
        except Exception as e:
            logger.error(f"Error analyzing environmental impact: {str(e)}")
            return {}

    def analyze_safety_risks(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze safety and risks (Objective 4)."""
        try:
            results = {}
            
            # Analyze high-value shipments
            value_threshold = df['VALUE'].quantile(0.95)
            high_value_shipments = df[df['VALUE'] >= value_threshold]
            
            risk_analysis = {
                'high_value_routes': high_value_shipments.groupby('route')['VALUE'].sum().nlargest(10).to_dict(),
                'high_value_modes': high_value_shipments.groupby('DISAGMOT')['VALUE'].sum().to_dict()
            }
            
            results['risk_analysis'] = risk_analysis
            return results
        except Exception as e:
            logger.error(f"Error analyzing safety risks: {str(e)}")
            return {}

    def analyze_economic_disruptions(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze economic disruptions (Objective 5)."""
        try:
            results = {}
            
            # Analyze value trends over time
            value_trends = df.groupby('Date').agg({
                'VALUE': 'sum',
                'SHIPWT': 'sum'
            }).round(2)
            
            # Convert datetime index to string
            value_trends.index = value_trends.index.astype(str)
            results['value_trends'] = value_trends.to_dict()
            
            # Analyze trade balance
            trade_balance = df.groupby(['Date', 'TRDTYPE']).agg({
                'VALUE': 'sum'
            }).round(2)
            
            # Convert multi-index to string
            trade_balance_dict = {}
            for idx, val in trade_balance['VALUE'].items():
                key = f"{idx[0]}_{idx[1]}"
                trade_balance_dict[key] = val
            results['trade_balance'] = trade_balance_dict
            
            return results
        except Exception as e:
            logger.error(f"Error analyzing economic disruptions: {str(e)}")
            return {}

    def generate_recommendations(self, all_results: Dict[str, Any]) -> List[str]:
        """Generate data-driven recommendations (Objective 6)."""
        recommendations = []
        
        try:
            # Movement pattern recommendations
            if 'transport_mode_analysis' in all_results:
                mode_data = all_results['transport_mode_analysis']
                recommendations.append("Optimize modal split based on value density analysis")
            
            # Efficiency recommendations
            if 'mode_efficiency' in all_results:
                efficiency_data = all_results['mode_efficiency']
                recommendations.append("Focus on improving efficiency in high-cost routes")
            
            # Environmental recommendations
            if 'emissions_by_mode' in all_results:
                emissions_data = all_results['emissions_by_mode']
                recommendations.append("Consider shifting to lower-emission transport modes where feasible")
            
            # Add more specific recommendations based on analysis results
            
        except Exception as e:
            logger.error(f"Error generating recommendations: {str(e)}")
        
        return recommendations

    def analyze_year(self, year: str) -> Dict[str, Any]:
        """Analyze data for a specific year."""
        try:
            # Load data
            file_path = self.output_dir / f'freight_data_{year}_processed.parquet'
            if not file_path.exists():
                logger.error(f"Data file not found: {file_path}")
                return {}
            
            df = pd.read_parquet(file_path)
            
            # Run analyses
            results = {
                'year': year,
                'freight_movements': self.analyze_freight_movements(df),
                'operational_efficiency': self.analyze_operational_efficiency(df),
                'environmental_impact': self.analyze_environmental_impact(df),
                'safety_risks': self.analyze_safety_risks(df),
                'economic_disruptions': self.analyze_economic_disruptions(df)
            }
            
            # Generate recommendations
            results['recommendations'] = self.generate_recommendations(results)
            
            # Save results
            output_file = self.results_dir / f'analysis_{year}.json'
            with open(output_file, 'w') as f:
                json.dump(results, f, indent=2)
            
            return results
            
        except Exception as e:
            logger.error(f"Error analyzing year {year}: {str(e)}")
            return {}

def main():
    """Main function to run the analysis."""
    base_dir = Path(__file__).parent.parent
    analyzer = FreightAnalysis(base_dir / 'output')
    
    all_years_results = {}
    for year in range(2020, 2025):
        logger.info(f"Analyzing year {year}")
        year_results = analyzer.analyze_year(str(year))
        if year_results:
            all_years_results[str(year)] = year_results
    
    # Save combined results
    output_file = base_dir / 'output' / 'all_years_analysis.json'
    with open(output_file, 'w') as f:
        json.dump(all_years_results, f, indent=2)
    logger.info("Analysis complete!")

if __name__ == "__main__":
    main()
