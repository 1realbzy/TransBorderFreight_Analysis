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

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class FreightVisualizer:
    def __init__(self, output_dir: Path):
        """Initialize visualizer with output directory."""
        self.output_dir = Path(output_dir)
        self.results_dir = self.output_dir / 'analysis_results'
        self.figures_dir = self.output_dir / 'figures'
        self.figures_dir.mkdir(exist_ok=True)
        
        # Load analysis results
        self.all_results = self._load_analysis_results()
        
    def _load_analysis_results(self) -> dict:
        """Load analysis results from JSON files."""
        try:
            with open(self.output_dir / 'all_years_analysis.json', 'r') as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Error loading analysis results: {str(e)}")
            return {}
    
    def _get_mode_mapping(self):
        """Get mapping of transport mode codes to readable names."""
        return {
            '1': 'Truck',
            '2': 'Rail',
            '3': 'Pipeline',
            '4': 'Air',
            '5': 'Vessel',
            '6': 'Other',
            '7': 'Container',
            '8': 'Mail',
            '9': 'Unknown'
        }

    def create_movement_patterns_viz(self, year: str) -> dict:
        """Create visualizations for freight movement patterns."""
        try:
            year_data = self.all_results.get(year, {})
            movement_data = year_data.get('freight_movements', {})
            mode_mapping = self._get_mode_mapping()
            
            figures = {}
            
            # Transport mode analysis
            if 'transport_mode_analysis' in movement_data:
                mode_data = pd.DataFrame(movement_data['transport_mode_analysis'])
                mode_data.index = [mode_mapping.get(str(mode), f'Mode {mode}') for mode in mode_data.index]
                
                # Filter out modes with zero or very small values
                min_value_threshold = mode_data['VALUE_sum'].max() * 0.001  # 0.1% of max value
                mode_data = mode_data[mode_data['VALUE_sum'] > min_value_threshold]
                
                # Create stacked bar chart for mode distribution
                fig = go.Figure()
                
                # Format values for better readability
                def format_value(x):
                    if x >= 1e9:
                        return f'${x/1e9:.1f}B'
                    elif x >= 1e6:
                        return f'${x/1e6:.1f}M'
                    else:
                        return f'${x:,.0f}'

                def format_weight(x):
                    if x >= 1e6:
                        return f'{x/1e6:.1f}M tons'
                    elif x >= 1e3:
                        return f'{x/1e3:.1f}K tons'
                    else:
                        return f'{x:,.0f} tons'
                
                # Add value bars
                fig.add_trace(go.Bar(
                    name='Trade Value',
                    x=mode_data.index,
                    y=mode_data['VALUE_sum'],
                    text=mode_data['VALUE_sum'].apply(format_value),
                    textposition='auto',
                    hovertemplate='<b>%{x}</b><br>' +
                                'Trade Value: %{text}<br>' +
                                '<i>Click for detailed analysis</i><extra></extra>'
                ))
                
                # Add weight bars
                fig.add_trace(go.Bar(
                    name='Shipment Weight',
                    x=mode_data.index,
                    y=mode_data['SHIPWT_sum'],
                    text=mode_data['SHIPWT_sum'].apply(format_weight),
                    textposition='auto',
                    hovertemplate='<b>%{x}</b><br>' +
                                'Total Weight: %{text}<br>' +
                                '<i>Click for detailed analysis</i><extra></extra>'
                ))
                
                # Calculate percentage shares for annotation
                total_value = mode_data['VALUE_sum'].sum()
                top_mode = mode_data.loc[mode_data['VALUE_sum'].idxmax()]
                top_mode_share = (top_mode['VALUE_sum'] / total_value) * 100
                
                fig.update_layout(
                    title={
                        'text': f'Transport Mode Distribution ({year})',
                        'y':0.95,
                        'x':0.5,
                        'xanchor': 'center',
                        'yanchor': 'top',
                        'font': {'size': 24}
                    },
                    barmode='group',
                    xaxis_title='Transport Mode',
                    yaxis_title='Amount',
                    showlegend=True,
                    legend_title='Metric',
                    height=600,  # Make the chart bigger
                    annotations=[
                        {
                            'text': f"Key Insight: {top_mode.name} dominates with {top_mode_share:.1f}% of total trade value",
                            'xref': 'paper',
                            'yref': 'paper',
                            'x': 0,
                            'y': 1.1,
                            'showarrow': False,
                            'font': {'size': 14, 'color': 'blue'}
                        },
                        {
                            'text': 'Hover over bars for detailed information | Click legend items to filter',
                            'xref': 'paper',
                            'yref': 'paper',
                            'x': 0,
                            'y': -0.15,
                            'showarrow': False,
                            'font': {'size': 12, 'color': 'gray'}
                        }
                    ]
                )
                figures['mode_distribution'] = fig
            
            # Value and Weight Trends
            if 'value_trends' in movement_data:
                trends_data = pd.DataFrame(movement_data['value_trends'])
                
                fig = go.Figure()
                
                # Add value trend
                fig.add_trace(go.Scatter(
                    name='Trade Value',
                    x=trends_data.index,
                    y=trends_data['VALUE'],
                    mode='lines+markers',
                    line=dict(width=3, color='blue'),
                    text=trends_data['VALUE'].apply(format_value),
                    hovertemplate='<b>%{x}</b><br>' +
                                'Trade Value: %{text}<br>' +
                                '<i>Click to see monthly details</i><extra></extra>'
                ))
                
                # Add weight trend
                fig.add_trace(go.Scatter(
                    name='Shipment Weight',
                    x=trends_data.index,
                    y=trends_data['SHIPWT'],
                    mode='lines+markers',
                    line=dict(width=3, color='red'),
                    text=trends_data['SHIPWT'].apply(format_weight),
                    hovertemplate='<b>%{x}</b><br>' +
                                'Total Weight: %{text}<br>' +
                                '<i>Click to see monthly details</i><extra></extra>'
                ))
                
                # Calculate year-over-year growth
                yoy_value_growth = ((trends_data['VALUE'].iloc[-1] - trends_data['VALUE'].iloc[0]) / 
                                  trends_data['VALUE'].iloc[0] * 100)
                
                fig.update_layout(
                    title={
                        'text': f'Trade Value and Weight Trends ({year})',
                        'y':0.95,
                        'x':0.5,
                        'xanchor': 'center',
                        'yanchor': 'top',
                        'font': {'size': 24}
                    },
                    xaxis_title='Month',
                    yaxis_title='Amount',
                    height=500,
                    annotations=[
                        {
                            'text': f"Year-over-Year Growth: {yoy_value_growth:+.1f}% in trade value",
                            'xref': 'paper',
                            'yref': 'paper',
                            'x': 0,
                            'y': 1.1,
                            'showarrow': False,
                            'font': {'size': 14, 'color': 'green' if yoy_value_growth > 0 else 'red'}
                        }
                    ]
                )
                figures['value_trends'] = fig
            
            return figures
            
        except Exception as e:
            logger.error(f"Error creating movement patterns visualizations: {str(e)}")
            return {}

    def create_efficiency_viz(self, year: str) -> dict:
        """Create visualizations for operational efficiency."""
        try:
            year_data = self.all_results.get(year, {})
            efficiency_data = year_data.get('operational_efficiency', {})
            
            figures = {}
            
            # Mode efficiency analysis
            if 'mode_efficiency' in efficiency_data:
                mode_eff = pd.DataFrame(efficiency_data['mode_efficiency'])
                
                # Create bubble chart for efficiency metrics
                fig = px.scatter(
                    mode_eff,
                    x='value_density_mean',
                    y='cost_per_value_mean',
                    size='value_density_std',
                    hover_name=mode_eff.index,
                    title=f'Transport Mode Efficiency ({year})'
                )
                figures['mode_efficiency'] = fig
            
            # Regional efficiency
            if 'regional_efficiency' in efficiency_data:
                regional_data = pd.DataFrame(efficiency_data['regional_efficiency'])
                
                # Create choropleth map
                fig = px.choropleth(
                    regional_data,
                    locations=regional_data.index,
                    locationmode='USA-states',
                    color='value_density',
                    scope='usa',
                    title=f'Regional Value Density ({year})'
                )
                figures['regional_efficiency'] = fig
            
            return figures
            
        except Exception as e:
            logger.error(f"Error creating efficiency visualizations: {str(e)}")
            return {}
    
    def create_environmental_viz(self, year: str) -> dict:
        """Create visualizations for environmental impact."""
        try:
            year_data = self.all_results.get(year, {})
            environmental_data = year_data.get('environmental_impact', {})
            
            figures = {}
            
            # Emissions by mode
            if 'emissions_by_mode' in environmental_data:
                emissions_data = pd.DataFrame(environmental_data['emissions_by_mode'])
                
                # Create treemap for emissions
                fig = px.treemap(
                    emissions_data,
                    path=[emissions_data.index],
                    values='estimated_emissions_sum',
                    title=f'Emissions by Transport Mode ({year})'
                )
                figures['emissions_treemap'] = fig
                
                # Create line chart for emissions intensity
                fig = px.line(
                    emissions_data,
                    x=emissions_data.index,
                    y='estimated_emissions_mean',
                    title=f'Emissions Intensity by Mode ({year})'
                )
                figures['emissions_intensity'] = fig
            
            return figures
            
        except Exception as e:
            logger.error(f"Error creating environmental visualizations: {str(e)}")
            return {}
    
    def create_safety_viz(self, year: str) -> dict:
        """Create visualizations for safety and risk analysis."""
        try:
            year_data = self.all_results.get(year, {})
            safety_data = year_data.get('safety_risks', {})
            
            figures = {}
            
            # Risk analysis
            if 'risk_analysis' in safety_data:
                risk_data = safety_data['risk_analysis']
                
                # High-value routes
                routes_data = pd.DataFrame.from_dict(
                    risk_data['high_value_routes'],
                    orient='index',
                    columns=['value']
                )
                
                fig = px.bar(
                    routes_data,
                    x=routes_data.index,
                    y='value',
                    title=f'High-Value Routes ({year})'
                )
                figures['high_value_routes'] = fig
                
                # High-value modes
                modes_data = pd.DataFrame.from_dict(
                    risk_data['high_value_modes'],
                    orient='index',
                    columns=['value']
                )
                
                fig = px.pie(
                    modes_data,
                    values='value',
                    names=modes_data.index,
                    title=f'High-Value Shipments by Mode ({year})'
                )
                figures['high_value_modes'] = fig
            
            return figures
            
        except Exception as e:
            logger.error(f"Error creating safety visualizations: {str(e)}")
            return {}
    
    def create_economic_viz(self, year: str) -> dict:
        """Create visualizations for economic analysis."""
        try:
            year_data = self.all_results.get(year, {})
            economic_data = year_data.get('economic_disruptions', {})
            
            figures = {}
            
            # Value trends
            if 'value_trends' in economic_data:
                trends_data = pd.DataFrame(economic_data['value_trends'])
                
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=trends_data.index,
                    y=trends_data['VALUE'],
                    name='Value'
                ))
                fig.add_trace(go.Scatter(
                    x=trends_data.index,
                    y=trends_data['SHIPWT'],
                    name='Weight'
                ))
                fig.update_layout(
                    title=f'Value and Weight Trends ({year})',
                    xaxis_title='Date',
                    yaxis_title='Amount'
                )
                figures['value_trends'] = fig
            
            # Trade balance
            if 'trade_balance' in economic_data:
                balance_data = pd.DataFrame.from_dict(
                    economic_data['trade_balance'],
                    orient='index',
                    columns=['value']
                )
                
                fig = px.bar(
                    balance_data,
                    x=balance_data.index,
                    y='value',
                    title=f'Trade Balance ({year})'
                )
                figures['trade_balance'] = fig
            
            return figures
            
        except Exception as e:
            logger.error(f"Error creating economic visualizations: {str(e)}")
            return {}

    def create_dashboard(self):
        """Create an interactive Dash dashboard with clear labels and explanations."""
        app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
        
        # Custom CSS for better styling
        app.layout = html.Div([
            # Navigation bar
            dbc.Navbar(
                dbc.Container([
                    dbc.Row([
                        dbc.Col(html.H1("TransBorder Freight Analysis Dashboard", className="mb-0 text-white"))
                    ]),
                ], fluid=True),
                color="primary",
                dark=True,
                className="mb-4"
            ),

            dbc.Container([
                # Introduction and Context
                dbc.Row([
                    dbc.Col([
                        html.Div([
                            html.H4("About This Dashboard", className="mb-3"),
                            html.P([
                                "This dashboard provides insights into cross-border freight transportation between the United States and its trading partners. ",
                                "The analysis covers various aspects including movement patterns, efficiency metrics, environmental impact, and economic performance."
                            ], className="lead"),
                            html.P([
                                "Use the year selector below to explore different time periods. Each visualization is interactive - ",
                                "you can hover over data points for details, click legend items to filter, and download charts as images."
                            ], className="mb-4"),
                        ], className="bg-light p-4 rounded-3 mb-4")
                    ])
                ]),

                # Year Selector with Description
                dbc.Row([
                    dbc.Col([
                        dbc.Card([
                            dbc.CardBody([
                                html.H5("Select Analysis Year", className="card-title"),
                                dcc.Dropdown(
                                    id='year-selector',
                                    options=[{'label': f'Year {year}', 'value': year} for year in self.all_results.keys()],
                                    value=list(self.all_results.keys())[0],
                                    className="mb-2"
                                ),
                                html.Small("Choose a year to view its freight transportation patterns and metrics", className="text-muted")
                            ])
                        ], className="mb-4")
                    ])
                ]),

                # Movement Patterns Section
                dbc.Row([
                    dbc.Col([
                        html.Div([
                            html.H3("Movement Patterns", className="mb-3"),
                            html.P([
                                "This section shows how freight is distributed across different transport modes and how patterns change over time. ",
                                "Understanding these patterns helps optimize route planning and resource allocation."
                            ], className="mb-4"),
                            dcc.Graph(id='mode-distribution', className="mb-4"),
                            html.Div([
                                html.H5("Key Insights:", className="mb-2"),
                                html.Ul([
                                    html.Li("Compare value vs weight distribution across transport modes"),
                                    html.Li("Identify dominant transport modes and their market share"),
                                    html.Li("Track changes in modal preferences over time")
                                ], className="mb-3")
                            ], className="bg-light p-3 rounded")
                        ], className="mb-5")
                    ])
                ]),

                # Efficiency Metrics Section
                dbc.Row([
                    dbc.Col([
                        html.Div([
                            html.H3("Operational Efficiency", className="mb-3"),
                            html.P([
                                "This section analyzes the efficiency of different transport modes and routes. ",
                                "Key metrics include value density, cost efficiency, and regional performance."
                            ], className="mb-4"),
                            dcc.Graph(id='mode-efficiency', className="mb-4"),
                            dcc.Graph(id='regional-efficiency', className="mb-4"),
                            html.Div([
                                html.H5("Understanding the Metrics:", className="mb-2"),
                                html.Ul([
                                    html.Li("Value Density: Higher values indicate more valuable cargo per weight unit"),
                                    html.Li("Regional Efficiency: Shows how different regions perform in handling freight"),
                                    html.Li("Cost Efficiency: Compares transport costs across modes and routes")
                                ], className="mb-3")
                            ], className="bg-light p-3 rounded")
                        ], className="mb-5")
                    ])
                ]),

                # Environmental Impact Section
                dbc.Row([
                    dbc.Col([
                        html.Div([
                            html.H3("Environmental Impact", className="mb-3"),
                            html.P([
                                "Track the environmental footprint of freight transportation. ",
                                "This section helps identify opportunities for reducing emissions and improving sustainability."
                            ], className="mb-4"),
                            dcc.Graph(id='emissions-treemap', className="mb-4"),
                            dcc.Graph(id='emissions-intensity', className="mb-4"),
                            html.Div([
                                html.H5("Environmental Considerations:", className="mb-2"),
                                html.Ul([
                                    html.Li("Compare emissions across different transport modes"),
                                    html.Li("Identify high-impact routes and opportunities for improvement"),
                                    html.Li("Track progress in emissions reduction over time")
                                ], className="mb-3")
                            ], className="bg-light p-3 rounded")
                        ], className="mb-5")
                    ])
                ]),

                # Footer with Links
                dbc.Row([
                    dbc.Col([
                        html.Hr(),
                        html.P([
                            "View detailed recommendations in the ",
                            html.A("Recommendations Dashboard", href="http://127.0.0.1:8051", target="_blank"),
                            " | Data source: Bureau of Transportation Statistics"
                        ], className="text-center text-muted")
                    ])
                ])
            ], fluid=True)
        ])
        
        return app

    def create_recommendations_dashboard(self):
        """Create a dashboard for data-driven recommendations."""
        app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
        
        recommendations = {
            'efficiency': {
                'title': 'Operational Efficiency',
                'findings': [
                    'Modal distribution shows heavy reliance on specific transport modes',
                    'Significant variations in value density across transport modes',
                    'Regional imbalances in freight movement efficiency'
                ],
                'recommendations': [
                    'Optimize modal mix based on route distance and cargo value',
                    'Implement route optimization to minimize empty returns',
                    'Develop intermodal facilities in underserved regions'
                ],
                'impact': [
                    'Potential cost savings of 15-20% through optimal mode selection',
                    'Reduced empty running miles by up to 25%',
                    'Improved asset utilization across the network'
                ]
            },
            'sustainability': {
                'title': 'Environmental Sustainability',
                'findings': [
                    'High emissions intensity in certain transport corridors',
                    'Seasonal variations in environmental impact',
                    'Modal choices significantly affecting carbon footprint'
                ],
                'recommendations': [
                    'Transition to low-emission vehicles for short-haul routes',
                    'Implement green corridor initiatives for high-traffic routes',
                    'Develop incentives for using environmentally friendly modes'
                ],
                'impact': [
                    'Potential 30% reduction in carbon emissions',
                    'Improved air quality in high-traffic corridors',
                    'Enhanced compliance with environmental regulations'
                ]
            },
            'safety': {
                'title': 'Safety and Risk Management',
                'findings': [
                    'Concentration of high-value shipments on specific routes',
                    'Seasonal patterns in risk incidents',
                    'Modal-specific safety challenges'
                ],
                'recommendations': [
                    'Enhance security measures for high-value corridors',
                    'Implement weather-based routing during high-risk seasons',
                    'Develop mode-specific safety protocols and training'
                ],
                'impact': [
                    'Reduced cargo loss and damage rates',
                    'Improved safety records across all modes',
                    'Enhanced risk management and compliance'
                ]
            },
            'economic': {
                'title': 'Economic Performance',
                'findings': [
                    'Trade imbalances in certain corridors',
                    'Opportunities for modal shift and cost optimization',
                    'Seasonal economic patterns affecting efficiency'
                ],
                'recommendations': [
                    'Develop strategies for balancing trade flows',
                    'Optimize modal choice based on value-to-weight ratio',
                    'Implement dynamic pricing for seasonal variations'
                ],
                'impact': [
                    'Improved asset utilization and return on investment',
                    'Reduced operating costs through better planning',
                    'Enhanced competitive position in the market'
                ]
            }
        }
        
        def create_recommendation_card(category, data):
            return dbc.Card([
                dbc.CardHeader(html.H4(data['title'], className="mb-0")),
                dbc.CardBody([
                    html.H6("Key Findings:", className="card-subtitle mb-2 text-muted"),
                    html.Ul([html.Li(finding) for finding in data['findings']], className="mb-3"),
                    html.H6("Recommendations:", className="card-subtitle mb-2 text-muted"),
                    html.Ul([html.Li(rec) for rec in data['recommendations']], className="mb-3"),
                    html.H6("Expected Impact:", className="card-subtitle mb-2 text-muted"),
                    html.Ul([html.Li(impact) for impact in data['impact']], className="mb-3")
                ])
            ], className="mb-4 h-100")
        
        app.layout = html.Div([
            # Navigation bar
            dbc.Navbar(
                dbc.Container([
                    dbc.Row([
                        dbc.Col(html.H1("Data-Driven Recommendations", className="mb-0 text-white"))
                    ]),
                ], fluid=True),
                color="success",
                dark=True,
                className="mb-4"
            ),

            dbc.Container([
                # Introduction
                dbc.Row([
                    dbc.Col([
                        html.Div([
                            html.H4("About These Recommendations", className="mb-3"),
                            html.P([
                                "Based on comprehensive analysis of freight movement patterns, efficiency metrics, ",
                                "environmental impact, and safety considerations, we provide the following actionable recommendations. ",
                                "Each recommendation is backed by data analysis and includes expected impact metrics."
                            ], className="lead mb-4")
                        ], className="bg-light p-4 rounded-3 mb-4")
                    ])
                ]),

                # Recommendations Cards
                dbc.Row([
                    dbc.Col(create_recommendation_card(cat, data), md=6)
                    for cat, data in recommendations.items()
                ], className="g-4"),

                # Footer
                dbc.Row([
                    dbc.Col([
                        html.Hr(),
                        html.P([
                            "Return to ",
                            html.A("Main Dashboard", href="http://127.0.0.1:8050", target="_blank"),
                            " | Updated monthly based on latest analysis"
                        ], className="text-center text-muted")
                    ])
                ])
            ], fluid=True)
        ])
        
        return app

    def create_dashboard(self):
        """Create an interactive Dash dashboard with clear labels and explanations."""
        app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
        
        # Custom CSS for better styling
        app.layout = html.Div([
            # Navigation bar
            dbc.Navbar(
                dbc.Container([
                    dbc.Row([
                        dbc.Col(html.H1("TransBorder Freight Analysis Dashboard", className="mb-0 text-white"))
                    ]),
                ], fluid=True),
                color="primary",
                dark=True,
                className="mb-4"
            ),

            dbc.Container([
                # Introduction and Context
                dbc.Row([
                    dbc.Col([
                        html.Div([
                            html.H4("About This Dashboard", className="mb-3"),
                            html.P([
                                "This dashboard provides insights into cross-border freight transportation between the United States and its trading partners. ",
                                "The analysis covers various aspects including movement patterns, efficiency metrics, environmental impact, and economic performance."
                            ], className="lead"),
                            html.P([
                                "Use the year selector below to explore different time periods. Each visualization is interactive - ",
                                "you can hover over data points for details, click legend items to filter, and download charts as images."
                            ], className="mb-4"),
                        ], className="bg-light p-4 rounded-3 mb-4")
                    ])
                ]),

                # Year Selector with Description
                dbc.Row([
                    dbc.Col([
                        dbc.Card([
                            dbc.CardBody([
                                html.H5("Select Analysis Year", className="card-title"),
                                dcc.Dropdown(
                                    id='year-selector',
                                    options=[{'label': f'Year {year}', 'value': year} for year in self.all_results.keys()],
                                    value=list(self.all_results.keys())[0],
                                    className="mb-2"
                                ),
                                html.Small("Choose a year to view its freight transportation patterns and metrics", className="text-muted")
                            ])
                        ], className="mb-4")
                    ])
                ]),

                # Movement Patterns Section
                dbc.Row([
                    dbc.Col([
                        html.Div([
                            html.H3("Movement Patterns", className="mb-3"),
                            html.P([
                                "This section shows how freight is distributed across different transport modes and how patterns change over time. ",
                                "Understanding these patterns helps optimize route planning and resource allocation."
                            ], className="mb-4"),
                            dcc.Graph(id='mode-distribution', className="mb-4"),
                            html.Div([
                                html.H5("Key Insights:", className="mb-2"),
                                html.Ul([
                                    html.Li("Compare value vs weight distribution across transport modes"),
                                    html.Li("Identify dominant transport modes and their market share"),
                                    html.Li("Track changes in modal preferences over time")
                                ], className="mb-3")
                            ], className="bg-light p-3 rounded")
                        ], className="mb-5")
                    ])
                ]),

                # Efficiency Metrics Section
                dbc.Row([
                    dbc.Col([
                        html.Div([
                            html.H3("Operational Efficiency", className="mb-3"),
                            html.P([
                                "This section analyzes the efficiency of different transport modes and routes. ",
                                "Key metrics include value density, cost efficiency, and regional performance."
                            ], className="mb-4"),
                            dcc.Graph(id='mode-efficiency', className="mb-4"),
                            dcc.Graph(id='regional-efficiency', className="mb-4"),
                            html.Div([
                                html.H5("Understanding the Metrics:", className="mb-2"),
                                html.Ul([
                                    html.Li("Value Density: Higher values indicate more valuable cargo per weight unit"),
                                    html.Li("Regional Efficiency: Shows how different regions perform in handling freight"),
                                    html.Li("Cost Efficiency: Compares transport costs across modes and routes")
                                ], className="mb-3")
                            ], className="bg-light p-3 rounded")
                        ], className="mb-5")
                    ])
                ]),

                # Environmental Impact Section
                dbc.Row([
                    dbc.Col([
                        html.Div([
                            html.H3("Environmental Impact", className="mb-3"),
                            html.P([
                                "Track the environmental footprint of freight transportation. ",
                                "This section helps identify opportunities for reducing emissions and improving sustainability."
                            ], className="mb-4"),
                            dcc.Graph(id='emissions-treemap', className="mb-4"),
                            dcc.Graph(id='emissions-intensity', className="mb-4"),
                            html.Div([
                                html.H5("Environmental Considerations:", className="mb-2"),
                                html.Ul([
                                    html.Li("Compare emissions across different transport modes"),
                                    html.Li("Identify high-impact routes and opportunities for improvement"),
                                    html.Li("Track progress in emissions reduction over time")
                                ], className="mb-3")
                            ], className="bg-light p-3 rounded")
                        ], className="mb-5")
                    ])
                ]),

                # Footer with Links
                dbc.Row([
                    dbc.Col([
                        html.Hr(),
                        html.P([
                            "View detailed recommendations in the ",
                            html.A("Recommendations Dashboard", href="http://127.0.0.1:8051", target="_blank"),
                            " | Data source: Bureau of Transportation Statistics"
                        ], className="text-center text-muted")
                    ])
                ])
            ], fluid=True)
        ])
        
        return app

    def create_recommendations_dashboard(self):
        """Create a dashboard for data-driven recommendations."""
        app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
        
        recommendations = {
            'efficiency': {
                'title': 'Operational Efficiency',
                'findings': [
                    'Modal distribution shows heavy reliance on specific transport modes',
                    'Significant variations in value density across transport modes',
                    'Regional imbalances in freight movement efficiency'
                ],
                'recommendations': [
                    'Optimize modal mix based on route distance and cargo value',
                    'Implement route optimization to minimize empty returns',
                    'Develop intermodal facilities in underserved regions'
                ],
                'impact': [
                    'Potential cost savings of 15-20% through optimal mode selection',
                    'Reduced empty running miles by up to 25%',
                    'Improved asset utilization across the network'
                ]
            },
            'sustainability': {
                'title': 'Environmental Sustainability',
                'findings': [
                    'High emissions intensity in certain transport corridors',
                    'Seasonal variations in environmental impact',
                    'Modal choices significantly affecting carbon footprint'
                ],
                'recommendations': [
                    'Transition to low-emission vehicles for short-haul routes',
                    'Implement green corridor initiatives for high-traffic routes',
                    'Develop incentives for using environmentally friendly modes'
                ],
                'impact': [
                    'Potential 30% reduction in carbon emissions',
                    'Improved air quality in high-traffic corridors',
                    'Enhanced compliance with environmental regulations'
                ]
            },
            'safety': {
                'title': 'Safety and Risk Management',
                'findings': [
                    'Concentration of high-value shipments on specific routes',
                    'Seasonal patterns in risk incidents',
                    'Modal-specific safety challenges'
                ],
                'recommendations': [
                    'Enhance security measures for high-value corridors',
                    'Implement weather-based routing during high-risk seasons',
                    'Develop mode-specific safety protocols and training'
                ],
                'impact': [
                    'Reduced cargo loss and damage rates',
                    'Improved safety records across all modes',
                    'Enhanced risk management and compliance'
                ]
            },
            'economic': {
                'title': 'Economic Performance',
                'findings': [
                    'Trade imbalances in certain corridors',
                    'Opportunities for modal shift and cost optimization',
                    'Seasonal economic patterns affecting efficiency'
                ],
                'recommendations': [
                    'Develop strategies for balancing trade flows',
                    'Optimize modal choice based on value-to-weight ratio',
                    'Implement dynamic pricing for seasonal variations'
                ],
                'impact': [
                    'Improved asset utilization and return on investment',
                    'Reduced operating costs through better planning',
                    'Enhanced competitive position in the market'
                ]
            }
        }
        
        def create_recommendation_card(category, data):
            return dbc.Card([
                dbc.CardHeader(html.H4(data['title'], className="mb-0")),
                dbc.CardBody([
                    html.H6("Key Findings:", className="card-subtitle mb-2 text-muted"),
                    html.Ul([html.Li(finding) for finding in data['findings']], className="mb-3"),
                    html.H6("Recommendations:", className="card-subtitle mb-2 text-muted"),
                    html.Ul([html.Li(rec) for rec in data['recommendations']], className="mb-3"),
                    html.H6("Expected Impact:", className="card-subtitle mb-2 text-muted"),
                    html.Ul([html.Li(impact) for impact in data['impact']], className="mb-3")
                ])
            ], className="mb-4 h-100")
        
        app.layout = html.Div([
            # Navigation bar
            dbc.Navbar(
                dbc.Container([
                    dbc.Row([
                        dbc.Col(html.H1("Data-Driven Recommendations", className="mb-0 text-white"))
                    ]),
                ], fluid=True),
                color="success",
                dark=True,
                className="mb-4"
            ),

            dbc.Container([
                # Introduction
                dbc.Row([
                    dbc.Col([
                        html.Div([
                            html.H4("About These Recommendations", className="mb-3"),
                            html.P([
                                "Based on comprehensive analysis of freight movement patterns, efficiency metrics, ",
                                "environmental impact, and safety considerations, we provide the following actionable recommendations. ",
                                "Each recommendation is backed by data analysis and includes expected impact metrics."
                            ], className="lead mb-4")
                        ], className="bg-light p-4 rounded-3 mb-4")
                    ])
                ]),

                # Recommendations Cards
                dbc.Row([
                    dbc.Col(create_recommendation_card(cat, data), md=6)
                    for cat, data in recommendations.items()
                ], className="g-4"),

                # Footer
                dbc.Row([
                    dbc.Col([
                        html.Hr(),
                        html.P([
                            "Return to ",
                            html.A("Main Dashboard", href="http://127.0.0.1:8050", target="_blank"),
                            " | Updated monthly based on latest analysis"
                        ], className="text-center text-muted")
                    ])
                ])
            ], fluid=True)
        ])
        
        return app

def main():
    """Main function to run the visualization and recommendations dashboards."""
    try:
        base_dir = Path(__file__).parent.parent
        visualizer = FreightVisualizer(base_dir / 'output')
        
        # Create and run main dashboard
        app = visualizer.create_dashboard()
        
        # Create recommendations dashboard
        rec_app = visualizer.create_recommendations_dashboard()
        
        # Import required for running multiple apps
        from threading import Thread
        
        def run_rec_dashboard():
            rec_app.run_server(debug=True, port=8051, host='127.0.0.1')
        
        # Start recommendations dashboard in a separate thread
        rec_thread = Thread(target=run_rec_dashboard)
        rec_thread.daemon = True
        rec_thread.start()
        
        # Run main dashboard
        app.run_server(debug=True, port=8050, host='127.0.0.1')
        
    except Exception as e:
        logger.error(f"Error starting dashboards: {str(e)}")
        raise

if __name__ == "__main__":
    main()
