"""
Create interactive dashboards for freight transportation analysis using Plotly
"""

import json
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import dash
from dash import dcc, html
import dash_bootstrap_components as dbc
from dash.dependencies import Input, Output
from pathlib import Path
import logging
import numpy as np
from threading import Thread

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class FreightVisualizer:
    def __init__(self, output_dir: Path):
        """Initialize visualizer with output directory."""
        self.output_dir = Path(output_dir)
        self.results_dir = self.output_dir / 'analysis_results'
        self.results_dir.mkdir(exist_ok=True)
        
        # Load analysis results
        self.all_results = self._load_analysis_results()

    def _load_analysis_results(self):
        """Load analysis results from JSON files."""
        try:
            with open(self.output_dir / 'all_years_analysis.json', 'r') as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Error loading analysis results: {str(e)}")
            return {}

    def create_main_dashboard(self):
        """Create the main analysis dashboard."""
        app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
        
        app.layout = dbc.Container([
            dbc.Row([
                dbc.Col([
                    html.H1("TransBorder Freight Analysis Dashboard", className="text-center mb-4"),
                    html.P(
                        "Analysis of cross-border freight transportation patterns from 2020 to 2024",
                        className="text-center mb-4"
                    ),
                ])
            ]),
            
            dbc.Row([
                dbc.Col([
                    dcc.Dropdown(
                        id='year-selector',
                        options=[{'label': f'Year {year}', 'value': year} 
                                for year in self.all_results.keys()],
                        value=list(self.all_results.keys())[0] if self.all_results else None,
                        className="mb-4"
                    )
                ])
            ]),
            
            dbc.Row([
                dbc.Col([
                    html.H3("Transport Mode Distribution", className="mb-3"),
                    dcc.Graph(id='mode-distribution')
                ])
            ]),
            
            dbc.Row([
                dbc.Col([
                    html.H3("Efficiency Metrics", className="mb-3"),
                    dcc.Graph(id='efficiency-metrics')
                ])
            ]),
            
            dbc.Row([
                dbc.Col([
                    html.H3("Environmental Impact", className="mb-3"),
                    dcc.Graph(id='environmental-impact')
                ])
            ]),
            
            dbc.Row([
                dbc.Col([
                    html.Hr(),
                    html.P([
                        "View detailed recommendations in the ",
                        html.A("Recommendations Dashboard", href="http://127.0.0.1:8051", target="_blank")
                    ], className="text-center")
                ])
            ])
        ], fluid=True)

        @app.callback(
            [Output('mode-distribution', 'figure'),
             Output('efficiency-metrics', 'figure'),
             Output('environmental-impact', 'figure')],
            [Input('year-selector', 'value')]
        )
        def update_graphs(selected_year):
            if not selected_year or selected_year not in self.all_results:
                return [go.Figure() for _ in range(3)]
            
            data = self.all_results[selected_year]
            
            # Mode Distribution
            mode_dist = go.Figure(data=[
                go.Bar(
                    x=list(data['movement_patterns']['mode_distribution'].keys()),
                    y=list(data['movement_patterns']['mode_distribution'].values()),
                    name='Value Share'
                )
            ])
            mode_dist.update_layout(title='Transport Mode Distribution')
            
            # Efficiency Metrics
            efficiency = go.Figure(data=[
                go.Bar(
                    x=list(data['efficiency_metrics']['mode_efficiency'].keys()),
                    y=list(data['efficiency_metrics']['mode_efficiency'].values()),
                    name='Efficiency Score'
                )
            ])
            efficiency.update_layout(title='Mode Efficiency Comparison')
            
            # Environmental Impact
            environmental = go.Figure(data=[
                go.Bar(
                    x=list(data['environmental_impact']['emissions_by_mode'].keys()),
                    y=list(data['environmental_impact']['emissions_by_mode'].values()),
                    name='Emissions'
                )
            ])
            environmental.update_layout(title='Environmental Impact by Mode')
            
            return mode_dist, efficiency, environmental
        
        return app

    def create_recommendations_dashboard(self):
        """Create the recommendations dashboard."""
        app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP], url_base_pathname='/recommendations/')
        
        recommendations = {
            'efficiency': {
                'title': 'Operational Efficiency',
                'findings': [
                    'Modal distribution shows heavy reliance on truck transport',
                    'Significant variations in value density across modes',
                    'Regional imbalances in freight movement efficiency'
                ],
                'recommendations': [
                    'Optimize modal mix based on distance and cargo value',
                    'Implement route optimization to minimize empty returns',
                    'Develop intermodal facilities in underserved regions'
                ]
            },
            'environmental': {
                'title': 'Environmental Sustainability',
                'findings': [
                    'High emissions intensity in certain corridors',
                    'Seasonal variations in environmental impact',
                    'Modal choices significantly affecting carbon footprint'
                ],
                'recommendations': [
                    'Transition to low-emission vehicles for short-haul routes',
                    'Implement green corridor initiatives',
                    'Develop incentives for eco-friendly modes'
                ]
            }
        }
        
        app.layout = dbc.Container([
            dbc.Row([
                dbc.Col([
                    html.H1("Freight Analysis Recommendations", className="text-center mb-4"),
                    html.P(
                        "Data-driven recommendations based on freight movement analysis",
                        className="text-center mb-4"
                    )
                ])
            ]),
            
            *[
                dbc.Row([
                    dbc.Col([
                        dbc.Card([
                            dbc.CardHeader(html.H3(data['title'])),
                            dbc.CardBody([
                                html.H5("Key Findings"),
                                html.Ul([html.Li(finding) for finding in data['findings']]),
                                html.H5("Recommendations"),
                                html.Ul([html.Li(rec) for rec in data['recommendations']])
                            ])
                        ], className="mb-4")
                    ])
                ])
                for category, data in recommendations.items()
            ],
            
            dbc.Row([
                dbc.Col([
                    html.Hr(),
                    html.P([
                        "Return to ",
                        html.A("Main Dashboard", href="http://127.0.0.1:8050", target="_blank")
                    ], className="text-center")
                ])
            ])
        ], fluid=True)
        
        return app

def run_dashboard(app, port):
    """Run a dashboard on specified port."""
    app.run_server(debug=False, port=port)

def main():
    """Run both dashboards."""
    try:
        visualizer = FreightVisualizer(Path("output"))
        
        # Create both apps
        main_app = visualizer.create_main_dashboard()
        rec_app = visualizer.create_recommendations_dashboard()
        
        # Create and start threads for both apps
        main_thread = Thread(target=run_dashboard, args=(main_app, 8050))
        rec_thread = Thread(target=run_dashboard, args=(rec_app, 8051))
        
        main_thread.daemon = True
        rec_thread.daemon = True
        
        main_thread.start()
        rec_thread.start()
        
        print("Main dashboard running at http://127.0.0.1:8050")
        print("Recommendations dashboard running at http://127.0.0.1:8051")
        
        # Keep the main thread alive
        try:
            while True:
                import time
                time.sleep(1)
        except KeyboardInterrupt:
            print("Shutting down dashboards...")
            
    except Exception as e:
        logger.error(f"Error running dashboards: {str(e)}")
        raise

if __name__ == "__main__":
    main()
