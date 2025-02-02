"""
Comprehensive Analysis of TransBorder Freight Data
"""
import pandas as pd
import numpy as np
from pathlib import Path
import plotly.express as px
import plotly.graph_objects as go
from detailed_patterns_analysis import DetailedPatternsAnalysis
from commodity_codes import COMMODITY_CODES, INDUSTRY_CATEGORIES

class ComprehensiveAnalysis:
    def __init__(self, data_dir: Path):
        self.data_dir = data_dir
        self.analysis = DetailedPatternsAnalysis(data_dir)
        self.output_dir = data_dir.parent / "output" / "detailed_patterns"
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def analyze_commodities(self, year: str):
        """Detailed analysis of commodity patterns"""
        print(f"\n=== Commodity Analysis for {year} ===")
        print("-" * 50)

        # Get commodity data
        results = self.analysis.analyze_commodity_distribution(year)
        df = results['commodity_data']

        # Add commodity descriptions
        df['commodity_desc'] = df['DEPE'].apply(lambda x: COMMODITY_CODES.get(x, 'Other'))
        df['industry_category'] = df['DEPE'].str[:2].map(INDUSTRY_CATEGORIES)

        # Industry-level analysis
        industry_stats = df.groupby('industry_category').agg({
            'VALUE': 'sum',
            'SHIPWT': 'sum'
        }).reset_index()

        print("\nIndustry-Level Analysis:")
        total_value = industry_stats['VALUE'].sum()
        for _, row in industry_stats.iterrows():
            if pd.notna(row['industry_category']):
                print(f"\n{row['industry_category']}:")
                print(f"- Total Value: ${row['VALUE']:,.2f}")
                print(f"- Share of Trade: {(row['VALUE']/total_value)*100:.1f}%")

        # Value density analysis
        df['value_density'] = df['VALUE'] / df['SHIPWT']
        top_density = df.nlargest(5, 'value_density')

        print("\nHighest Value Density Commodities:")
        for _, row in top_density.iterrows():
            print(f"\n{row['commodity_desc']} ({row['DEPE']}):")
            print(f"- Value per Weight Unit: ${row['value_density']:,.2f}")
            print(f"- Total Value: ${row['VALUE']:,.2f}")

    def analyze_trade_corridors(self, year: str):
        """Detailed analysis of trade corridors"""
        print(f"\n=== Trade Corridor Analysis for {year} ===")
        print("-" * 50)

        # Get corridor data
        results = self.analysis.analyze_trade_corridors(year)
        df = results['corridor_data']

        # Analyze by direction
        direction_stats = df.groupby('direction').agg({
            'VALUE': 'sum',
            'SHIPWT': 'sum'
        }).reset_index()

        print("\nTrade Balance Analysis:")
        total_value = direction_stats['VALUE'].sum()
        for _, row in direction_stats.iterrows():
            print(f"\n{row['direction']}:")
            print(f"- Total Value: ${row['VALUE']:,.2f}")
            print(f"- Share of Trade: {(row['VALUE']/total_value)*100:.1f}%")

        # Analyze top corridors
        top_corridors = df.nlargest(5, 'VALUE')
        print("\nTop Trade Corridors:")
        for _, row in top_corridors.iterrows():
            print(f"\n{row['origin']} → {row['destination']}:")
            print(f"- Total Value: ${row['VALUE']:,.2f}")
            print(f"- Total Weight: {row['SHIPWT']:,.2f}")
            print(f"- Value Density: ${row['VALUE']/row['SHIPWT']:,.2f} per unit")

    def analyze_seasonal_patterns(self, year: str):
        """Detailed analysis of seasonal patterns"""
        print(f"\n=== Seasonal Pattern Analysis for {year} ===")
        print("-" * 50)

        # Get seasonal data
        results = self.analysis.analyze_seasonal_patterns(year)
        df = results['seasonal_data']

        # Monthly trends
        print("\nMonthly Trends:")
        for _, row in df.iterrows():
            print(f"\n{row['month_name']}:")
            print(f"- Total Value: ${row['total_value']:,.2f}")
            print(f"- Shipments: {row['num_shipments']:,}")
            print(f"- Avg Value/Shipment: ${row['avg_value_per_shipment']:,.2f}")

        # Quarterly analysis
        df['quarter'] = pd.to_numeric(df['month']).apply(lambda x: (x-1)//3 + 1)
        quarterly = df.groupby('quarter').agg({
            'total_value': 'sum',
            'num_shipments': 'sum'
        }).reset_index()

        print("\nQuarterly Analysis:")
        for _, row in quarterly.iterrows():
            print(f"\nQ{int(row['quarter'])}:")
            print(f"- Total Value: ${row['total_value']:,.2f}")
            print(f"- Total Shipments: {row['num_shipments']:,}")

        # Seasonality indicators
        cv = df['total_value'].std() / df['total_value'].mean()
        print(f"\nSeasonality Coefficient of Variation: {cv:.2f}")
        
        # Peak-to-Trough ratio
        peak = df['total_value'].max()
        trough = df['total_value'].min()
        print(f"Peak-to-Trough Ratio: {peak/trough:.2f}")

    def run_comprehensive_analysis(self, year: str):
        """Run all analyses"""
        self.analyze_commodities(year)
        self.analyze_trade_corridors(year)
        self.analyze_seasonal_patterns(year)

if __name__ == "__main__":
    data_dir = Path(__file__).parent.parent / "data"
    analysis = ComprehensiveAnalysis(data_dir)
    analysis.run_comprehensive_analysis('2020')
