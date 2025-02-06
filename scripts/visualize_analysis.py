"""
Create interactive dashboards for freight transportation analysis using Plotly
"""

import json
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from pathlib import Path
import logging
from typing import Dict, List
import numpy as np

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class FreightVisualizer:
    def __init__(self, analysis_dir: Path, output_dir: Path):
        self.analysis_dir = analysis_dir
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Load analysis results
        self.results = {}
        self._load_analysis_results()
        
    def _load_analysis_results(self) -> None:
        """Load all analysis results from JSON files"""
        try:
            for analysis_file in self.analysis_dir.glob("*_analysis.json"):
                logger.info(f"Loading {analysis_file.name}")
                try:
                    with open(analysis_file, 'r') as f:
                        key = analysis_file.stem.replace('_analysis', '')
                        self.results[key] = json.load(f)
                    logger.info(f"Successfully loaded {analysis_file.name}")
                except Exception as e:
                    logger.error(f"Error loading {analysis_file.name}: {str(e)}")
                    continue
            
            if not self.results:
                raise ValueError("No analysis files were loaded successfully")
                
            logger.info(f"Loaded {len(self.results)} analysis files")
            logger.info(f"Available analyses: {list(self.results.keys())}")
            
        except Exception as e:
            logger.error(f"Error in data loading: {str(e)}")
            raise

    def create_movement_patterns_dashboard(self) -> None:
        """Create dashboard for freight movement patterns"""
        try:
            if 'freight_patterns' not in self.results:
                logger.error("Freight patterns analysis results not found")
                return
                
            patterns = self.results['freight_patterns']
            logger.info("Creating movement patterns dashboard...")
            
            # Create figure with subplots
            fig = make_subplots(
                rows=2, cols=2,
                subplot_titles=(
                    'Volume by Transport Mode',
                    'Regional Distribution',
                    'Seasonal Patterns',
                    'Route Complexity'
                )
            )
            
            # 1. Volume by Transport Mode
            try:
                mode_trends = pd.DataFrame(patterns['mode_trends'])
                mode_trends['Date'] = pd.to_datetime(mode_trends['Date'])
                for mode in mode_trends['DISAGMOT'].unique():
                    mode_data = mode_trends[mode_trends['DISAGMOT'] == mode]
                    fig.add_trace(
                        go.Scatter(
                            x=mode_data['Date'],
                            y=mode_data['VALUE'],
                            name=f'Mode {mode}',
                            mode='lines'
                        ),
                        row=1, col=1
                    )
                logger.info("Added volume trends")
            except Exception as e:
                logger.error(f"Error creating volume trends: {str(e)}")
            
            # 2. Regional Distribution
            try:
                regional = pd.DataFrame(patterns['regional_distribution'])
                top_regions = regional.nlargest(10, 'VALUE')
                fig.add_trace(
                    go.Bar(
                        x=top_regions['USASTATE'],
                        y=top_regions['VALUE'],
                        name='Top 10 States'
                    ),
                    row=1, col=2
                )
                logger.info("Added regional distribution")
            except Exception as e:
                logger.error(f"Error creating regional distribution: {str(e)}")
            
            # 3. Seasonal Patterns
            try:
                seasonal = pd.DataFrame(patterns['seasonal_patterns'])
                for mode in seasonal['DISAGMOT'].unique():
                    mode_data = seasonal[seasonal['DISAGMOT'] == mode]
                    fig.add_trace(
                        go.Scatter(
                            x=mode_data['Month'],
                            y=mode_data['VALUE'],
                            name=f'Mode {mode} (Seasonal)',
                            mode='lines+markers'
                        ),
                        row=2, col=1
                    )
                logger.info("Added seasonal patterns")
            except Exception as e:
                logger.error(f"Error creating seasonal patterns: {str(e)}")
            
            # 4. Route Complexity
            try:
                route = pd.DataFrame(patterns['route_analysis'])
                fig.add_trace(
                    go.Box(
                        x=route['DISAGMOT'],
                        y=route['VALUE'],
                        name='Value Distribution'
                    ),
                    row=2, col=2
                )
                logger.info("Added route complexity")
            except Exception as e:
                logger.error(f"Error creating route complexity: {str(e)}")
            
            # Update layout
            fig.update_layout(
                height=1000,
                width=1600,
                showlegend=True,
                title_text="Freight Movement Patterns Dashboard"
            )
            
            # Save dashboard
            output_path = self.output_dir / "movement_patterns_dashboard.html"
            fig.write_html(str(output_path))
            logger.info(f"Saved movement patterns dashboard to {output_path}")
            
        except Exception as e:
            logger.error(f"Error creating movement patterns dashboard: {str(e)}")
            raise

    def create_operational_efficiency_dashboard(self) -> None:
        """Create dashboard for operational efficiency"""
        try:
            if 'operational_efficiency' not in self.results:
                logger.error("Operational efficiency analysis results not found")
                return
                
            efficiency = self.results['operational_efficiency']
            logger.info("Creating operational efficiency dashboard...")
            
            # Create figure with subplots
            fig = make_subplots(
                rows=2, cols=2,
                subplot_titles=(
                    'Value Density Trends',
                    'Transport Mode Efficiency',
                    'Regional Throughput',
                    'Efficiency Metrics'
                )
            )
            
            # 1. Value Density Trends
            try:
                density = pd.DataFrame(efficiency['value_density'])
                density['Date'] = pd.to_datetime(density['Date'])
                for mode in density['DISAGMOT'].unique():
                    mode_data = density[density['DISAGMOT'] == mode]
                    fig.add_trace(
                        go.Scatter(
                            x=mode_data['Date'],
                            y=mode_data['ValueDensity'],
                            name=f'Mode {mode}',
                            mode='lines'
                        ),
                        row=1, col=1
                    )
                logger.info("Added value density trends")
            except Exception as e:
                logger.error(f"Error creating value density trends: {str(e)}")
            
            # 2. Transport Mode Efficiency
            try:
                mode_eff = pd.DataFrame(efficiency['mode_efficiency'])
                fig.add_trace(
                    go.Bar(
                        x=mode_eff['DISAGMOT'],
                        y=mode_eff['ValuePerWeight'],
                        name='Value per Weight'
                    ),
                    row=1, col=2
                )
                logger.info("Added transport mode efficiency")
            except Exception as e:
                logger.error(f"Error creating transport mode efficiency: {str(e)}")
            
            # 3. Regional Throughput
            try:
                throughput = pd.DataFrame(efficiency['regional_throughput'])
                throughput['Date'] = pd.to_datetime(throughput['Date'])
                fig.add_trace(
                    go.Scatter(
                        x=throughput['Date'],
                        y=throughput['Throughput'],
                        mode='lines',
                        name='Overall Throughput'
                    ),
                    row=2, col=1
                )
                logger.info("Added regional throughput")
            except Exception as e:
                logger.error(f"Error creating regional throughput: {str(e)}")
            
            # 4. Efficiency Metrics
            try:
                metrics = pd.DataFrame(efficiency['efficiency_metrics'])
                fig.add_trace(
                    go.Indicator(
                        mode="number+gauge+delta",
                        value=metrics['LoadFactor'].mean(),
                        domain={'x': [0, 0.5], 'y': [0, 0.5]},
                        title={'text': "Load Factor (%)"},
                        gauge={'axis': {'range': [0, 100]}},
                        delta={'reference': metrics['LoadFactor'].shift(1).mean()}
                    ),
                    row=2, col=2
                )
                
                fig.add_trace(
                    go.Indicator(
                        mode="number+gauge+delta",
                        value=metrics['TurnoverRate'].mean(),
                        domain={'x': [0.6, 1], 'y': [0, 0.5]},
                        title={'text': "Turnover Rate"},
                        delta={'reference': metrics['TurnoverRate'].shift(1).mean()}
                    ),
                    row=2, col=2
                )
                
                # Add time series of key metrics
                for metric in ['DeliveryTime', 'CostPerMile', 'UtilizationRate']:
                    fig.add_trace(
                        go.Scatter(
                            x=metrics['Date'],
                            y=metrics[metric],
                            name=metric,
                            mode='lines+markers'
                        ),
                        row=2, col=2
                    )
                logger.info("Added efficiency metrics")
            except Exception as e:
                logger.error(f"Error creating efficiency metrics: {str(e)}")
            
            # Update layout
            fig.update_layout(
                height=1000,
                width=1600,
                showlegend=True,
                title_text="Operational Efficiency Dashboard"
            )
            
            # Save dashboard
            output_path = self.output_dir / "operational_efficiency_dashboard.html"
            fig.write_html(str(output_path))
            logger.info(f"Saved operational efficiency dashboard to {output_path}")
            
        except Exception as e:
            logger.error(f"Error creating operational efficiency dashboard: {str(e)}")
            raise

    def create_environmental_impact_dashboard(self) -> None:
        """Create dashboard for environmental impact"""
        try:
            if 'environmental_impact' not in self.results:
                logger.error("Environmental impact analysis results not found")
                return
                
            environmental = self.results['environmental_impact']
            logger.info("Creating environmental impact dashboard...")
            
            # Create figure with subplots
            fig = make_subplots(
                rows=2, cols=2,
                subplot_titles=(
                    'Emissions by Transport Mode',
                    'Environmental Efficiency',
                    'Regional Environmental Impact',
                    'Emissions Trends'
                )
            )
            
            # 1. Emissions by Transport Mode
            try:
                emissions = pd.DataFrame(environmental['emissions_by_mode'])
                emissions['Date'] = pd.to_datetime(emissions['Date'])
                for mode in emissions['DISAGMOT'].unique():
                    mode_data = emissions[emissions['DISAGMOT'] == mode]
                    fig.add_trace(
                        go.Scatter(
                            x=mode_data['Date'],
                            y=mode_data['EstimatedEmissions'],
                            name=f'Mode {mode}',
                            mode='lines'
                        ),
                        row=1, col=1
                    )
                logger.info("Added emissions by transport mode")
            except Exception as e:
                logger.error(f"Error creating emissions by transport mode: {str(e)}")
            
            # 2. Environmental Efficiency
            try:
                env_eff = pd.DataFrame(environmental['environmental_efficiency'])
                fig.add_trace(
                    go.Bar(
                        x=env_eff['DISAGMOT'],
                        y=env_eff['EmissionsPerValue'],
                        name='Emissions per Value'
                    ),
                    row=1, col=2
                )
                logger.info("Added environmental efficiency")
            except Exception as e:
                logger.error(f"Error creating environmental efficiency: {str(e)}")
            
            # 3. Regional Environmental Impact
            try:
                regional_env = pd.DataFrame(environmental['regional_impact'])
                regional_env['Date'] = pd.to_datetime(regional_env['Date'])
                fig.add_trace(
                    go.Scatter(
                        x=regional_env['Date'],
                        y=regional_env['EmissionsPerTon'],
                        mode='lines',
                        name='Emissions per Ton'
                    ),
                    row=2, col=1
                )
                logger.info("Added regional environmental impact")
            except Exception as e:
                logger.error(f"Error creating regional environmental impact: {str(e)}")
            
            # 4. Emissions Trends
            try:
                trends = pd.DataFrame(environmental['emissions_trends'])
                
                # Overall emissions trend
                fig.add_trace(
                    go.Scatter(
                        x=trends['Date'],
                        y=trends['TotalEmissions'],
                        name='Total Emissions',
                        mode='lines',
                        line=dict(width=3)
                    ),
                    row=2, col=2
                )
                
                # Emissions by type
                for emission_type in ['CO2', 'NOx', 'PM25', 'SOx']:
                    fig.add_trace(
                        go.Scatter(
                            x=trends['Date'],
                            y=trends[emission_type],
                            name=f'{emission_type} Emissions',
                            mode='lines',
                            line=dict(dash='dot')
                        ),
                        row=2, col=2
                    )
                
                # Add year-over-year change
                fig.add_trace(
                    go.Bar(
                        x=trends['Date'],
                        y=trends['YoYChange'],
                        name='YoY Change (%)',
                        marker_color=trends['YoYChange'].apply(
                            lambda x: 'red' if x > 0 else 'green'
                        )
                    ),
                    row=2, col=2
                )
                logger.info("Added emissions trends")
            except Exception as e:
                logger.error(f"Error creating emissions trends: {str(e)}")
            
            # Update layout
            fig.update_layout(
                height=1000,
                width=1600,
                showlegend=True,
                title_text="Environmental Impact Dashboard"
            )
            
            # Save dashboard
            output_path = self.output_dir / "environmental_impact_dashboard.html"
            fig.write_html(str(output_path))
            logger.info(f"Saved environmental impact dashboard to {output_path}")
            
        except Exception as e:
            logger.error(f"Error creating environmental impact dashboard: {str(e)}")
            raise

    def create_economic_dashboard(self) -> None:
        """Create dashboard for economic analysis"""
        try:
            if 'economic_disruptions' not in self.results:
                logger.error("Economic disruptions analysis results not found")
                return
                
            economic = self.results['economic_disruptions']
            logger.info("Creating economic dashboard...")
            
            # Create figure with subplots
            fig = make_subplots(
                rows=2, cols=2,
                subplot_titles=(
                    'Monthly Trade Values',
                    'Growth Rates',
                    'Trade Patterns',
                    'Modal Shifts'
                )
            )
            
            # 1. Monthly Trade Values
            try:
                monthly = pd.DataFrame(economic['monthly_trends'])
                monthly['Date'] = pd.to_datetime(monthly['Date'])
                fig.add_trace(
                    go.Scatter(
                        x=monthly['Date'],
                        y=monthly['VALUE'],
                        name='Trade Value',
                        mode='lines'
                    ),
                    row=1, col=1
                )
                logger.info("Added monthly trade values")
            except Exception as e:
                logger.error(f"Error creating monthly trade values: {str(e)}")
            
            # 2. Growth Rates
            try:
                fig.add_trace(
                    go.Scatter(
                        x=monthly['Date'],
                        y=monthly['ValueGrowth'],
                        name='Value Growth (%)',
                        mode='lines'
                    ),
                    row=1, col=2
                )
                logger.info("Added growth rates")
            except Exception as e:
                logger.error(f"Error creating growth rates: {str(e)}")
            
            # 3. Trade Patterns
            try:
                patterns = pd.DataFrame(economic['trade_patterns'])
                patterns['Date'] = pd.to_datetime(patterns['Date'])
                for mode in patterns['DISAGMOT'].unique():
                    mode_data = patterns[patterns['DISAGMOT'] == mode]
                    fig.add_trace(
                        go.Scatter(
                            x=mode_data['Date'],
                            y=mode_data['VALUE'],
                            name=f'Mode {mode}',
                            mode='lines'
                        ),
                        row=2, col=1
                    )
                logger.info("Added trade patterns")
            except Exception as e:
                logger.error(f"Error creating trade patterns: {str(e)}")
            
            # 4. Modal Shifts Analysis
            try:
                shifts = pd.DataFrame(economic['modal_shifts'])
                
                # Modal shift patterns
                fig.add_trace(
                    go.Bar(
                        x=shifts['DISAGMOT'],
                        y=shifts['ShiftPercentage'],
                        name='Modal Shift (%)',
                        marker_color='lightblue'
                    ),
                    row=2, col=2
                )
                
                # Add shift drivers
                fig.add_trace(
                    go.Scatter(
                        x=shifts['Date'],
                        y=shifts['CostDriven'],
                        name='Cost-Driven Shifts',
                        mode='lines+markers',
                        line=dict(dash='dot')
                    ),
                    row=2, col=2
                )
                
                fig.add_trace(
                    go.Scatter(
                        x=shifts['Date'],
                        y=shifts['TimeDriven'],
                        name='Time-Driven Shifts',
                        mode='lines+markers',
                        line=dict(dash='dash')
                    ),
                    row=2, col=2
                )
                
                # Add resilience index
                fig.add_trace(
                    go.Scatter(
                        x=shifts['Date'],
                        y=shifts['ResilienceIndex'],
                        name='Modal Resilience Index',
                        mode='lines',
                        line=dict(width=2)
                    ),
                    row=2, col=2
                )
                logger.info("Added modal shifts analysis")
            except Exception as e:
                logger.error(f"Error creating modal shifts analysis: {str(e)}")
            
            # Update layout
            fig.update_layout(
                height=1000,
                width=1600,
                showlegend=True,
                title_text="Economic Analysis Dashboard"
            )
            
            # Save dashboard
            output_path = self.output_dir / "economic_dashboard.html"
            fig.write_html(str(output_path))
            logger.info(f"Saved economic dashboard to {output_path}")
            
        except Exception as e:
            logger.error(f"Error creating economic dashboard: {str(e)}")
            raise

    def create_all_dashboards(self) -> None:
        """Create all dashboards"""
        try:
            logger.info("Starting dashboard creation...")
            
            self.create_movement_patterns_dashboard()
            self.create_operational_efficiency_dashboard()
            self.create_environmental_impact_dashboard()
            self.create_economic_dashboard()
            
            logger.info("Completed creating all dashboards")
            
        except Exception as e:
            logger.error(f"Error creating dashboards: {str(e)}")
            raise

if __name__ == "__main__":
    try:
        # Set up directories
        base_dir = Path(__file__).parent.parent
        analysis_dir = base_dir / "output" / "analysis_results"
        viz_output_dir = base_dir / "output" / "dashboards"
        
        # Create and run visualizer
        visualizer = FreightVisualizer(analysis_dir, viz_output_dir)
        visualizer.create_all_dashboards()
        
    except Exception as e:
        logger.error(f"Script execution failed: {str(e)}")
        raise
