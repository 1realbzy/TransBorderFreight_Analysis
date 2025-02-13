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
        """Analyze freight movement patterns and trends (2020-2024)."""
        try:
            results = {}
            
            # 1. Transport mode analysis
            mode_stats = df.groupby(['DISAGMOT', pd.Grouper(key='Date', freq='M')]).agg({
                'VALUE': ['sum', 'mean'],
                'SHIPWT': ['sum', 'mean']
            }).round(2)
            
            # 2. Route analysis with congestion detection
            df['route'] = df['USASTATE'] + '-' + df['MEXSTATE'].fillna('') + df['CANPROV'].fillna('')
            route_analysis = df.groupby(['route', pd.Grouper(key='Date', freq='M')]).agg({
                'VALUE': 'sum',
                'SHIPWT': 'sum',
                'FREIGHT_CHARGES': 'mean'
            }).round(2)
            
            # 3. Global events impact analysis
            # Add pandemic period indicator
            df['period'] = pd.cut(df['Date'],
                                bins=[
                                    pd.Timestamp('2020-01-01'),
                                    pd.Timestamp('2020-03-01'),  # Pre-pandemic
                                    pd.Timestamp('2021-06-01'),  # Peak pandemic
                                    pd.Timestamp('2024-12-31')   # Post-pandemic
                                ],
                                labels=['pre_pandemic', 'peak_pandemic', 'post_pandemic'])
            
            pandemic_impact = df.groupby(['period', 'DISAGMOT']).agg({
                'VALUE': 'sum',
                'SHIPWT': 'sum'
            }).round(2)
            
            results.update({
                'mode_trends': mode_stats.to_dict(),
                'route_analysis': route_analysis.to_dict(),
                'pandemic_impact': pandemic_impact.to_dict()
            })
            
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
        """Analyze environmental impact and sustainability metrics."""
        try:
            results = {}
            
            # Define emission factors (kg CO2 per ton-mile)
            emission_factors = {
                'TRUCK': 0.161,      # EPA estimates
                'RAIL': 0.023,       # EPA estimates
                'AIR': 1.527,        # ICAO estimates
                'WATER': 0.048,      # IMO estimates
            }
            
            # Calculate emissions
            df['emissions'] = df.apply(
                lambda row: emission_factors.get(row['DISAGMOT'], 0) * row['SHIPWT'],
                axis=1
            )
            
            # Sustainability metrics by mode
            sustainability_metrics = df.groupby(['DISAGMOT', pd.Grouper(key='Date', freq='M')]).agg({
                'emissions': ['sum', 'mean'],
                'SHIPWT': 'sum',
                'VALUE': 'sum'
            }).round(2)
            
            # Calculate efficiency metrics
            sustainability_metrics['emissions_per_value'] = (
                sustainability_metrics[('emissions', 'sum')] / 
                sustainability_metrics[('VALUE', 'sum')]
            )
            
            results['sustainability_metrics'] = sustainability_metrics.to_dict()
            
            return results
        except Exception as e:
            logger.error(f"Error analyzing environmental impact: {str(e)}")
            return {}

    def analyze_safety_risks(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze safety incidents and risk factors."""
        try:
            results = {}
            
            # Safety incident analysis by mode and region
            safety_metrics = df.groupby(['DISAGMOT', 'USASTATE']).agg({
                'incident_count': 'sum',
                'SHIPWT': 'sum'
            }).round(2)
            
            # Calculate incident rates per 10000 shipments
            safety_metrics['incident_rate'] = (
                safety_metrics['incident_count'] / 
                safety_metrics['SHIPWT'] * 10000
            )
            
            # Identify high-risk areas
            high_risk_areas = safety_metrics[
                safety_metrics['incident_rate'] > 
                safety_metrics['incident_rate'].mean() + 
                safety_metrics['incident_rate'].std()
            ]
            
            results.update({
                'safety_metrics': safety_metrics.to_dict(),
                'high_risk_areas': high_risk_areas.to_dict()
            })
            
            return results
        except Exception as e:
            logger.error(f"Error analyzing safety risks: {str(e)}")
            return {}

    def analyze_infrastructure_utilization(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze infrastructure utilization and optimization opportunities."""
        try:
            results = {}
            
            # Calculate utilization metrics
            df['utilization_rate'] = df['SHIPWT'] / df['capacity'].fillna(df['SHIPWT'].max())
            
            # Infrastructure utilization by mode and region
            utilization_metrics = df.groupby(['DISAGMOT', 'USASTATE']).agg({
                'utilization_rate': ['mean', 'std'],
                'SHIPWT': 'sum',
                'capacity': 'mean'
            }).round(2)
            
            # Identify underutilized infrastructure
            underutilized = utilization_metrics[
                utilization_metrics[('utilization_rate', 'mean')] < 
                utilization_metrics[('utilization_rate', 'mean')].mean() - 
                utilization_metrics[('utilization_rate', 'std')]
            ]
            
            results.update({
                'utilization_metrics': utilization_metrics.to_dict(),
                'underutilized_infrastructure': underutilized.to_dict()
            })
            
            return results
        except Exception as e:
            logger.error(f"Error analyzing infrastructure utilization: {str(e)}")
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

    def analyze_delays_and_congestion(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze delays and congestion patterns with root cause analysis."""
        try:
            results = {}
            
            # Calculate average transit times by route
            df['transit_time'] = (pd.to_datetime(df['delivery_date']) - 
                                pd.to_datetime(df['shipment_date'])).dt.total_seconds() / 3600  # hours
            
            route_delays = df.groupby(['route', pd.Grouper(key='Date', freq='M')]).agg({
                'transit_time': ['mean', 'std'],
                'SHIPWT': 'sum'
            }).round(2)
            
            # Identify congested routes (above 75th percentile transit time)
            delay_threshold = df['transit_time'].quantile(0.75)
            congested_routes = df[df['transit_time'] > delay_threshold].groupby('route').agg({
                'transit_time': 'mean',
                'SHIPWT': 'sum',
                'VALUE': 'sum'
            }).round(2)
            
            # Analyze contributing factors
            delay_factors = df[df['transit_time'] > delay_threshold].groupby('delay_reason').agg({
                'transit_time': ['count', 'mean'],
                'VALUE': 'sum'
            }).round(2)
            
            results.update({
                'route_delays': route_delays.to_dict(),
                'congested_routes': congested_routes.to_dict(),
                'delay_factors': delay_factors.to_dict()
            })
            
            return results
        except Exception as e:
            logger.error(f"Error analyzing delays: {str(e)}")
            return {}

    def analyze_global_events_impact(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze impact of global events beyond pandemic."""
        try:
            results = {}
            
            # Define major events timeline
            events = {
                'trade_war': {
                    'start': '2020-01-01',
                    'end': '2020-12-31'
                },
                'supply_chain_crisis': {
                    'start': '2021-03-01',
                    'end': '2021-12-31'
                },
                'inflation_period': {
                    'start': '2022-01-01',
                    'end': '2022-12-31'
                }
            }
            
            # Analyze impact for each event
            event_impacts = {}
            for event, period in events.items():
                # Filter data for event period
                event_data = df[
                    (df['Date'] >= period['start']) & 
                    (df['Date'] <= period['end'])
                ]
                
                # Calculate impact metrics
                impact = event_data.groupby('DISAGMOT').agg({
                    'VALUE': ['sum', 'mean'],
                    'SHIPWT': ['sum', 'mean'],
                    'FREIGHT_CHARGES': 'mean'
                }).round(2)
                
                event_impacts[event] = impact.to_dict()
            
            results['event_impacts'] = event_impacts
            
            return results
        except Exception as e:
            logger.error(f"Error analyzing global events: {str(e)}")
            return {}

    def analyze_sustainability_scalability(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze scalability potential of sustainable transportation modes."""
        try:
            results = {}
            
            # Calculate sustainability scores
            mode_sustainability = df.groupby('DISAGMOT').agg({
                'emissions': ['mean', 'sum'],
                'SHIPWT': 'sum',
                'VALUE': 'sum',
                'capacity': 'mean'
            }).round(2)
            
            # Calculate efficiency metrics
            mode_sustainability['emissions_per_ton'] = (
                mode_sustainability[('emissions', 'sum')] / 
                mode_sustainability[('SHIPWT', 'sum')]
            )
            
            # Analyze capacity utilization trends
            capacity_trends = df.groupby(['DISAGMOT', pd.Grouper(key='Date', freq='M')]).agg({
                'utilization_rate': 'mean',
                'capacity': 'mean'
            }).round(2)
            
            # Calculate growth potential
            growth_potential = {}
            for mode in df['DISAGMOT'].unique():
                mode_data = df[df['DISAGMOT'] == mode]
                current_volume = mode_data['SHIPWT'].sum()
                max_capacity = mode_data['capacity'].max() * len(mode_data)
                growth_potential[mode] = {
                    'current_volume': current_volume,
                    'max_capacity': max_capacity,
                    'growth_percentage': ((max_capacity - current_volume) / current_volume * 100)
                }
            
            results.update({
                'mode_sustainability': mode_sustainability.to_dict(),
                'capacity_trends': capacity_trends.to_dict(),
                'growth_potential': growth_potential
            })
            
            return results
        except Exception as e:
            logger.error(f"Error analyzing sustainability scalability: {str(e)}")
            return {}

    def analyze_year_over_year_changes(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze year-over-year changes in key metrics."""
        try:
            results = {}
            
            # Add year and month columns
            df['year'] = pd.to_datetime(df['Date']).dt.year
            df['month'] = pd.to_datetime(df['Date']).dt.month
            
            # Calculate YoY changes for key metrics
            metrics = ['VALUE', 'SHIPWT', 'FREIGHT_CHARGES', 'emissions', 'utilization_rate']
            yoy_changes = {}
            
            for metric in metrics:
                # Monthly aggregation
                monthly_data = df.groupby(['year', 'month'])[metric].sum().reset_index()
                monthly_data['year_month'] = monthly_data['year'].astype(str) + '-' + monthly_data['month'].astype(str).str.zfill(2)
                
                # Calculate YoY change
                monthly_data['yoy_change'] = monthly_data.groupby('month')[metric].pct_change() * 100
                
                yoy_changes[metric] = monthly_data.set_index('year_month')['yoy_change'].to_dict()
            
            results['yoy_changes'] = yoy_changes
            
            # Seasonal patterns analysis
            seasonal_patterns = df.groupby(['year', 'month']).agg({
                'VALUE': 'sum',
                'SHIPWT': 'sum',
                'emissions': 'sum'
            }).round(2)
            
            results['seasonal_patterns'] = seasonal_patterns.to_dict()
            
            return results
        except Exception as e:
            logger.error(f"Error analyzing year-over-year changes: {str(e)}")
            return {}

    def analyze_mode_performance_metrics(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Calculate detailed performance metrics by transport mode."""
        try:
            results = {}
            
            # Efficiency metrics
            efficiency_metrics = df.groupby(['DISAGMOT', pd.Grouper(key='Date', freq='M')]).agg({
                'VALUE': 'sum',
                'SHIPWT': 'sum',
                'FREIGHT_CHARGES': 'sum',
                'transit_time': 'mean'
            }).round(2)
            
            # Calculate derived metrics
            efficiency_metrics['cost_per_ton'] = (
                efficiency_metrics['FREIGHT_CHARGES'] / 
                efficiency_metrics['SHIPWT']
            )
            efficiency_metrics['value_density'] = (
                efficiency_metrics['VALUE'] / 
                efficiency_metrics['SHIPWT']
            )
            efficiency_metrics['revenue_per_hour'] = (
                efficiency_metrics['VALUE'] / 
                efficiency_metrics['transit_time']
            )
            
            # Risk metrics
            risk_metrics = df.groupby('DISAGMOT').agg({
                'incident_count': ['sum', 'mean'],
                'SHIPWT': 'sum'
            }).round(2)
            
            risk_metrics['incident_rate'] = (
                risk_metrics[('incident_count', 'sum')] / 
                risk_metrics['SHIPWT'] * 1000000
            )  # incidents per million tons
            
            # Environmental metrics
            env_metrics = df.groupby('DISAGMOT').agg({
                'emissions': ['sum', 'mean'],
                'SHIPWT': 'sum',
                'distance': 'sum'
            }).round(2)
            
            env_metrics['emissions_per_ton_km'] = (
                env_metrics[('emissions', 'sum')] / 
                (env_metrics['SHIPWT'] * env_metrics['distance'])
            )
            
            results.update({
                'efficiency_metrics': efficiency_metrics.to_dict(),
                'risk_metrics': risk_metrics.to_dict(),
                'environmental_metrics': env_metrics.to_dict()
            })
            
            return results
        except Exception as e:
            logger.error(f"Error analyzing mode performance metrics: {str(e)}")
            return {}

    def analyze_route_optimization(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze route optimization opportunities."""
        try:
            results = {}
            
            # Route performance metrics
            route_metrics = df.groupby(['route', pd.Grouper(key='Date', freq='M')]).agg({
                'VALUE': 'sum',
                'SHIPWT': 'sum',
                'FREIGHT_CHARGES': 'sum',
                'transit_time': 'mean',
                'distance': 'mean',
                'utilization_rate': 'mean'
            }).round(2)
            
            # Calculate optimization metrics
            route_metrics['cost_per_ton_km'] = (
                route_metrics['FREIGHT_CHARGES'] / 
                (route_metrics['SHIPWT'] * route_metrics['distance'])
            )
            route_metrics['speed_kmh'] = (
                route_metrics['distance'] / 
                route_metrics['transit_time']
            )
            
            # Identify optimization opportunities
            avg_metrics = route_metrics.mean()
            optimization_opportunities = route_metrics[
                (route_metrics['cost_per_ton_km'] > avg_metrics['cost_per_ton_km']) &
                (route_metrics['utilization_rate'] < avg_metrics['utilization_rate'])
            ]
            
            results.update({
                'route_metrics': route_metrics.to_dict(),
                'optimization_opportunities': optimization_opportunities.to_dict()
            })
            
            return results
        except Exception as e:
            logger.error(f"Error analyzing route optimization: {str(e)}")
            return {}

    def generate_detailed_recommendations(self, all_results: Dict[str, Any]) -> Dict[str, List[str]]:
        """Generate detailed, data-driven recommendations by category."""
        try:
            recommendations = {
                'efficiency_improvements': [],
                'safety_enhancements': [],
                'environmental_sustainability': [],
                'infrastructure_optimization': [],
                'economic_resilience': []
            }
            
            # Efficiency recommendations
            if 'mode_performance_metrics' in all_results:
                metrics = all_results['mode_performance_metrics']
                for mode, data in metrics['efficiency_metrics'].items():
                    if data['cost_per_ton'] > metrics['efficiency_metrics']['cost_per_ton'].mean():
                        recommendations['efficiency_improvements'].append(
                            f"Optimize cost efficiency for {mode} transport - "
                            f"current cost per ton is {data['cost_per_ton']:.2f}, "
                            f"which is above average"
                        )
            
            # Safety recommendations
            if 'safety_risks' in all_results:
                risks = all_results['safety_risks']
                for area in risks['high_risk_areas']:
                    recommendations['safety_enhancements'].append(
                        f"Implement additional safety measures in {area} - "
                        f"incident rate is {risks['high_risk_areas'][area]['incident_rate']:.2f} "
                        f"per million shipments"
                    )
            
            # Environmental recommendations
            if 'environmental_impact' in all_results:
                env_data = all_results['environmental_impact']
                for mode, data in env_data['sustainability_metrics'].items():
                    if data['emissions_per_ton'] > env_data['sustainability_metrics']['emissions_per_ton'].mean():
                        recommendations['environmental_sustainability'].append(
                            f"Reduce emissions for {mode} transport - "
                            f"current emissions per ton are {data['emissions_per_ton']:.2f}, "
                            f"which is above average"
                        )
            
            # Infrastructure recommendations
            if 'infrastructure_utilization' in all_results:
                infra_data = all_results['infrastructure_utilization']
                for mode, data in infra_data['utilization_metrics'].items():
                    if data['utilization_rate'] < 0.7:  # Less than 70% utilization
                        recommendations['infrastructure_optimization'].append(
                            f"Improve utilization of {mode} infrastructure - "
                            f"current utilization rate is {data['utilization_rate']*100:.1f}%"
                        )
            
            # Economic recommendations
            if 'global_events' in all_results:
                events_data = all_results['global_events']
                for event, impact in events_data['event_impacts'].items():
                    if impact['VALUE']['mean'] < events_data['event_impacts']['VALUE']['mean'].mean():
                        recommendations['economic_resilience'].append(
                            f"Develop resilience strategies for {event.replace('_', ' ')} scenarios - "
                            f"observed {abs(impact['VALUE']['mean']):.1f}% impact on freight value"
                        )
            
            return recommendations
        except Exception as e:
            logger.error(f"Error generating detailed recommendations: {str(e)}")
            return {}

    def analyze_all(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Run all analyses and combine results."""
        try:
            return {
                'freight_movements': self.analyze_freight_movements(df),
                'delays_congestion': self.analyze_delays_and_congestion(df),
                'environmental_impact': self.analyze_environmental_impact(df),
                'safety_risks': self.analyze_safety_risks(df),
                'infrastructure_utilization': self.analyze_infrastructure_utilization(df),
                'global_events': self.analyze_global_events_impact(df),
                'sustainability_scalability': self.analyze_sustainability_scalability(df),
                'year_over_year': self.analyze_year_over_year_changes(df),
                'mode_performance': self.analyze_mode_performance_metrics(df),
                'route_optimization': self.analyze_route_optimization(df)
            }
        except Exception as e:
            logger.error(f"Error in complete analysis: {str(e)}")
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
                'infrastructure_utilization': self.analyze_infrastructure_utilization(df),
                'economic_disruptions': self.analyze_economic_disruptions(df),
                'delays_congestion': self.analyze_delays_and_congestion(df),
                'global_events': self.analyze_global_events_impact(df),
                'sustainability_scalability': self.analyze_sustainability_scalability(df),
                'year_over_year': self.analyze_year_over_year_changes(df),
                'mode_performance': self.analyze_mode_performance_metrics(df),
                'route_optimization': self.analyze_route_optimization(df)
            }
            
            # Generate recommendations
            results['recommendations'] = self.generate_detailed_recommendations(results)
            
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
