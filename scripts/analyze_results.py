"""
Results Analysis Script for TransBorder Freight Data
"""
import pandas as pd
import numpy as np
from pathlib import Path
from detailed_patterns_analysis import DetailedPatternsAnalysis

def analyze_2020_results():
    """Analyze the results from 2020 data"""
    # Initialize analysis
    data_dir = Path(__file__).parent.parent / "data"
    analysis = DetailedPatternsAnalysis(data_dir)
    
    # Get seasonal patterns
    seasonal_results = analysis.analyze_seasonal_patterns('2020')
    seasonal_df = seasonal_results['seasonal_data']
    
    print("\n=== 2020 TransBorder Freight Analysis Results ===")
    
    # Seasonal Analysis
    print("\n1. Seasonal Patterns:")
    print("-" * 50)
    
    # Find peak and low months
    peak_month = seasonal_df.loc[seasonal_df['total_value'].idxmax()]
    low_month = seasonal_df.loc[seasonal_df['total_value'].idxmin()]
    
    print(f"Peak Month: {peak_month['month_name']}")
    print(f"- Total Value: ${peak_month['total_value']:,.2f}")
    print(f"- Number of Shipments: {peak_month['num_shipments']:,}")
    print(f"- Average Value per Shipment: ${peak_month['avg_value_per_shipment']:,.2f}")
    
    print(f"\nLowest Month: {low_month['month_name']}")
    print(f"- Total Value: ${low_month['total_value']:,.2f}")
    print(f"- Number of Shipments: {low_month['num_shipments']:,}")
    print(f"- Average Value per Shipment: ${low_month['avg_value_per_shipment']:,.2f}")
    
    # Calculate quarterly totals
    seasonal_df['quarter'] = pd.to_numeric(seasonal_df['month']).apply(lambda x: (x-1)//3 + 1)
    quarterly = seasonal_df.groupby('quarter').agg({
        'total_value': 'sum',
        'num_shipments': 'sum'
    })
    
    print("\nQuarterly Analysis:")
    for q, row in quarterly.iterrows():
        print(f"Q{q} 2020:")
        print(f"- Total Value: ${row['total_value']:,.2f}")
        print(f"- Number of Shipments: {row['num_shipments']:,}")
    
    # Calculate year-over-year growth (if we had previous year's data)
    print("\nYear-over-Year Analysis:")
    print("Note: Previous year's data not available for comparison")
    
    # Trade Corridors Analysis
    corridor_results = analysis.analyze_trade_corridors('2020')
    corridor_data = corridor_results['corridor_data']
    
    print("\n2. Top Trade Corridors:")
    print("-" * 50)
    
    # Get top 5 corridors by value
    top_corridors = corridor_data.nlargest(5, 'VALUE')
    for _, row in top_corridors.iterrows():
        print(f"Origin: {row['origin']} → Destination: {row['destination']}")
        print(f"- Total Value: ${row['VALUE']:,.2f}")
        print(f"- Total Weight: {row['SHIPWT']:,.2f} units")
    
    # Commodity Analysis
    commodity_results = analysis.analyze_commodity_distribution('2020')
    commodity_data = commodity_results['commodity_data']
    
    print("\n3. Top Commodities:")
    print("-" * 50)
    
    # Get top 5 commodities by value
    top_commodities = commodity_data.nlargest(5, 'VALUE')
    for _, row in top_commodities.iterrows():
        print(f"Commodity Code: {row['DEPE']}")
        print(f"- Total Value: ${row['VALUE']:,.2f}")
        print(f"- Value Share: {row['value_percentage']:.1f}%")
        print(f"- Weight Share: {row['weight_percentage']:.1f}%")

if __name__ == "__main__":
    analyze_2020_results()
