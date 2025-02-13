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
from datetime import datetime
from typing import Dict, Any, List

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class FreightVisualizer:
    def __init__(self, results: Dict[str, Any]):
        """Initialize visualizer with analysis results."""
        self.results = results
        
    def create_dashboard(self):
        """Create a single dashboard with all visualizations."""
        app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
        
        app.layout = dbc.Container([
            dbc.Row([
                dbc.Col([
                    html.H1("Freight Transportation Analysis Dashboard",
                           className="text-center mb-4")
                ])
            ]),
            
            # Date Range Selector
            dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader("Select Date Range"),
                        dbc.CardBody([
                            dcc.DatePickerRange(
                                id='date-range',
                                start_date='2020-01-01',
                                end_date='2024-12-31',
                                min_date_allowed='2020-01-01',
                                max_date_allowed='2024-12-31'
                            )
                        ])
                    ])
                ], width=12)
            ], className="mb-4"),
            
            # Key Metrics
            dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader("Key Metrics"),
                        dbc.CardBody([
                            dcc.Graph(id="key-metrics-graph")
                        ])
                    ])
                ], width=12)
            ], className="mb-4"),
            
            # Mode Comparison
            dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader("Transport Mode Comparison"),
                        dbc.CardBody([
                            dcc.Dropdown(
                                id='metric-selector',
                                options=[
                                    {'label': 'Value', 'value': 'VALUE'},
                                    {'label': 'Weight', 'value': 'SHIPWT'},
                                    {'label': 'Emissions', 'value': 'emissions'},
                                    {'label': 'Cost per Ton', 'value': 'cost_per_ton'}
                                ],
                                value='VALUE'
                            ),
                            dcc.Graph(id="mode-comparison-graph")
                        ])
                    ])
                ], width=12)
            ], className="mb-4"),
            
            # Recommendations
            dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader("Recommendations"),
                        dbc.CardBody(id="recommendations-div")
                    ])
                ], width=12)
            ])
        ], fluid=True)
        
        @app.callback(
            [Output("key-metrics-graph", "figure"),
             Output("mode-comparison-graph", "figure"),
             Output("recommendations-div", "children")],
            [Input("date-range", "start_date"),
             Input("date-range", "end_date"),
             Input("metric-selector", "value")]
        )
        def update_graphs(start_date, end_date, selected_metric):
            # Key Metrics Graph
            key_fig = go.Figure()
            metrics_data = pd.DataFrame(self.results.get('freight_movements', {}).get('monthly_trends', {}))
            if not metrics_data.empty:
                key_fig.add_trace(go.Scatter(
                    x=metrics_data.index,
                    y=metrics_data['VALUE'],
                    name='Value'
                ))
            key_fig.update_layout(
                title="Monthly Freight Value Trends",
                xaxis_title="Date",
                yaxis_title="Value"
            )
            
            # Mode Comparison Graph
            mode_fig = go.Figure()
            mode_data = pd.DataFrame(self.results.get('mode_performance', {}).get('efficiency_metrics', {}))
            if not mode_data.empty:
                mode_fig.add_trace(go.Bar(
                    x=mode_data.index,
                    y=mode_data[selected_metric],
                    name=selected_metric
                ))
            mode_fig.update_layout(
                title=f"{selected_metric} by Transport Mode",
                xaxis_title="Transport Mode",
                yaxis_title=selected_metric
            )
            
            # Recommendations
            recommendations = self.results.get('recommendations', {})
            rec_div = []
            for category, recs in recommendations.items():
                rec_div.extend([
                    html.H5(category.replace('_', ' ').title()),
                    html.Ul([html.Li(rec) for rec in recs]),
                    html.Hr()
                ])
            
            return key_fig, mode_fig, rec_div
        
        return app

def main():
    """Main function to run the dashboard."""
    try:
        # Load sample data or use your actual data
        sample_results = {
            'freight_movements': {
                'monthly_trends': {
                    '2020-01': {'VALUE': 1000000},
                    '2020-02': {'VALUE': 1200000}
                }
            },
            'mode_performance': {
                'efficiency_metrics': {
                    'truck': {'VALUE': 500000, 'cost_per_ton': 100},
                    'rail': {'VALUE': 300000, 'cost_per_ton': 80}
                }
            },
            'recommendations': {
                'efficiency': ['Optimize truck routes', 'Increase rail utilization'],
                'environmental': ['Reduce emissions in high-traffic areas']
            }
        }
        
        visualizer = FreightVisualizer(sample_results)
        app = visualizer.create_dashboard()
        app.run_server(debug=True, port=8050)
        
    except Exception as e:
        print(f"Error running dashboard: {str(e)}")

if __name__ == '__main__':
    main()
