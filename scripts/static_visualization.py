import plotly.graph_objects as go
import plotly.io as pio
import plotly.subplots as sp
import pandas as pd
from pathlib import Path
import json

def create_visualizations(results_dir: Path):
    """Create comprehensive visualizations for all analysis objectives."""
    figures = []
    
    # Load analysis results
    with open(results_dir / 'all_years_analysis.json', 'r') as f:
        results = json.load(f)
    
    # 1. Freight Movement Trends (2020-2024)
    fig1 = sp.make_subplots(rows=2, cols=1, subplot_titles=['Value by Mode', 'Weight by Mode'])
    
    for mode in results['freight_movements']['mode_trends'].keys():
        fig1.add_trace(
            go.Scatter(
                x=list(results['freight_movements']['mode_trends'][mode]['dates']),
                y=list(results['freight_movements']['mode_trends'][mode]['values']),
                name=f'{mode} - Value',
                mode='lines'
            ),
            row=1, col=1
        )
        fig1.add_trace(
            go.Scatter(
                x=list(results['freight_movements']['mode_trends'][mode]['dates']),
                y=list(results['freight_movements']['mode_trends'][mode]['weights']),
                name=f'{mode} - Weight',
                mode='lines'
            ),
            row=2, col=1
        )
    
    fig1.update_layout(height=800, title_text="Freight Movement Trends (2020-2024)")
    figures.append(('freight_trends', fig1))
    
    # 2. Environmental Impact
    fig2 = sp.make_subplots(rows=1, cols=2, subplot_titles=['Emissions by Mode', 'Emissions per Ton-KM'])
    
    # Emissions by mode
    fig2.add_trace(
        go.Bar(
            x=list(results['environmental_impact']['mode_emissions'].keys()),
            y=list(results['environmental_impact']['mode_emissions'].values()),
            name='Total Emissions'
        ),
        row=1, col=1
    )
    
    # Emissions efficiency
    fig2.add_trace(
        go.Bar(
            x=list(results['environmental_impact']['emissions_efficiency'].keys()),
            y=list(results['environmental_impact']['emissions_efficiency'].values()),
            name='Emissions per Ton-KM'
        ),
        row=1, col=2
    )
    
    fig2.update_layout(height=500, title_text="Environmental Impact Analysis")
    figures.append(('environmental_impact', fig2))
    
    # 3. Safety Analysis
    fig3 = sp.make_subplots(rows=1, cols=2, subplot_titles=['Incident Rates by Mode', 'High Risk Areas'])
    
    fig3.add_trace(
        go.Bar(
            x=list(results['safety_risks']['incident_rates'].keys()),
            y=list(results['safety_risks']['incident_rates'].values()),
            name='Incident Rates'
        ),
        row=1, col=1
    )
    
    fig3.add_trace(
        go.Bar(
            x=list(results['safety_risks']['high_risk_areas'].keys()),
            y=[area['incident_count'] for area in results['safety_risks']['high_risk_areas'].values()],
            name='Incidents in High Risk Areas'
        ),
        row=1, col=2
    )
    
    fig3.update_layout(height=500, title_text="Safety Analysis")
    figures.append(('safety_analysis', fig3))
    
    # 4. Economic Impact
    fig4 = sp.make_subplots(rows=2, cols=1, subplot_titles=['Impact of Global Events', 'Economic Disruptions'])
    
    for event, data in results['global_events']['event_impacts'].items():
        fig4.add_trace(
            go.Scatter(
                x=list(data['dates']),
                y=list(data['impact_values']),
                name=event,
                mode='lines'
            ),
            row=1, col=1
        )
    
    for disruption, data in results['economic_disruptions']['disruption_impacts'].items():
        fig4.add_trace(
            go.Bar(
                x=[disruption],
                y=[data['value_impact']],
                name=disruption
            ),
            row=2, col=1
        )
    
    fig4.update_layout(height=800, title_text="Economic Impact Analysis")
    figures.append(('economic_impact', fig4))
    
    # 5. Infrastructure Utilization
    fig5 = go.Figure()
    
    for mode, data in results['infrastructure_utilization']['utilization_metrics'].items():
        fig5.add_trace(
            go.Scatter(
                x=list(data['dates']),
                y=list(data['utilization_rates']),
                name=mode,
                mode='lines'
            )
        )
    
    fig5.update_layout(
        title="Infrastructure Utilization Trends",
        xaxis_title="Date",
        yaxis_title="Utilization Rate (%)"
    )
    figures.append(('infrastructure', fig5))
    
    # Create output directory
    output_dir = Path('visualizations')
    output_dir.mkdir(exist_ok=True)
    
    # Save figures as HTML files
    for name, fig in figures:
        output_file = output_dir / f'{name}.html'
        pio.write_html(fig, file=str(output_file))
        print(f"Created visualization: {output_file}")
    
    # Create index.html with recommendations
    index_content = """
    <html>
    <head>
        <title>Freight Analysis Visualizations</title>
        <style>
            body { font-family: Arial, sans-serif; margin: 20px; max-width: 1200px; margin: 0 auto; padding: 20px; }
            h1, h2 { color: #333; }
            .viz-section { margin: 20px 0; padding: 20px; background: #f8f9fa; border-radius: 5px; }
            .viz-link { 
                display: inline-block;
                margin: 10px 0;
                padding: 10px 20px;
                background: #007bff;
                color: white;
                text-decoration: none;
                border-radius: 5px;
            }
            .viz-link:hover { background: #0056b3; }
            .recommendations { background: #e9ecef; padding: 20px; border-radius: 5px; margin-top: 20px; }
        </style>
    </head>
    <body>
        <h1>Freight Transportation Analysis (2020-2024)</h1>
        
        <div class="viz-section">
            <h2>Interactive Visualizations</h2>
    """
    
    for name, _ in figures:
        title = name.replace('_', ' ').title()
        index_content += f'<p><a class="viz-link" href="{name}.html">{title}</a></p>\n'
    
    # Add recommendations
    index_content += """
        </div>
        <div class="recommendations">
            <h2>Key Findings and Recommendations</h2>
    """
    
    for category, recs in results.get('recommendations', {}).items():
        index_content += f"<h3>{category.replace('_', ' ').title()}</h3><ul>\n"
        for rec in recs:
            index_content += f"<li>{rec}</li>\n"
        index_content += "</ul>\n"
    
    index_content += """
        </div>
    </body>
    </html>
    """
    
    with open(output_dir / 'index.html', 'w') as f:
        f.write(index_content)
    
    print(f"\nCreated visualization index at: {output_dir}/index.html")
    print("Open this file in your web browser to view all visualizations and recommendations")

if __name__ == '__main__':
    results_dir = Path('output')
    create_visualizations(results_dir)
