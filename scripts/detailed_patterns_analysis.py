"""
Detailed Patterns Analysis
-------------------------
This module focuses on analyzing:
1. Volume and routing trends over time
2. Seasonal patterns
3. Trade corridors (origin-destination pairs)
4. Commodity distribution
"""

import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
from typing import Dict, List
import calendar
from tqdm import tqdm

class DetailedPatternsAnalysis:
    def __init__(self, base_dir: str):
        """
        Initialize the detailed patterns analysis
        Args:
            base_dir: Base directory containing the data
        """
        self.base_dir = Path(base_dir)
        self.results_dir = self.base_dir.parent / "output" / "detailed_patterns"
        self.results_dir.mkdir(exist_ok=True, parents=True)
    
    def _process_file(self, file_path: Path) -> pd.DataFrame:
        """Process a single CSV file"""
        try:
            # Try different encodings
            encodings = ['utf-8', 'latin1', 'iso-8859-1']
            df = None
            
            for encoding in encodings:
                try:
                    df = pd.read_csv(file_path, dtype={
                        'TRDTYPE': str,
                        'USASTATE': str,
                        'DEPE': str,
                        'DISAGMOT': str,
                        'MEXSTATE': str,
                        'CANPROV': str,
                        'COUNTRY': str,
                        'VALUE': float,
                        'SHIPWT': float,
                        'FREIGHT_CHARGES': float,
                        'DF': str,
                        'CONTCODE': str,
                        'MONTH': str,
                        'YEAR': str
                    }, encoding=encoding)
                    break
                except UnicodeDecodeError:
                    continue
            
            if df is None:
                raise ValueError("Could not read file with any encoding")
            
            # Clean the data
            numeric_cols = ['VALUE', 'SHIPWT', 'FREIGHT_CHARGES']
            for col in numeric_cols:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
            
            return df
            
        except Exception as e:
            print(f"Error processing {file_path}: {str(e)}")
            return pd.DataFrame()
    
    def analyze_seasonal_patterns(self, year: str = '2020') -> Dict:
        """Analyze seasonal patterns within a year"""
        print(f"\nAnalyzing seasonal patterns for {year}...")
        
        # Get all CSV files for the specified year
        year_path = self.base_dir / year
        all_files = []
        for month_dir in year_path.glob("*/*/*.csv"):
            all_files.append(month_dir)
        
        # Process files
        monthly_data = []
        for file in tqdm(all_files, desc="Processing files"):
            df = self._process_file(file)
            if not df.empty:
                # Extract month from filename or path
                month = int(df['MONTH'].iloc[0])
                monthly_stats = {
                    'month': month,
                    'month_name': calendar.month_name[month],
                    'total_value': df['VALUE'].sum(),
                    'total_weight': df['SHIPWT'].sum(),
                    'num_shipments': len(df)
                }
                monthly_data.append(monthly_stats)
        
        # Convert to DataFrame
        seasonal_df = pd.DataFrame(monthly_data)
        
        # Create visualization
        fig = go.Figure()
        
        # Add value line
        fig.add_trace(go.Scatter(
            x=seasonal_df['month_name'],
            y=seasonal_df['total_value'],
            name='Total Value',
            line=dict(color='blue')
        ))
        
        # Add shipment count line
        fig.add_trace(go.Scatter(
            x=seasonal_df['month_name'],
            y=seasonal_df['num_shipments'],
            name='Number of Shipments',
            line=dict(color='red'),
            yaxis='y2'
        ))
        
        fig.update_layout(
            title=f'Seasonal Patterns in {year}',
            xaxis_title='Month',
            yaxis_title='Total Value',
            yaxis2=dict(
                title='Number of Shipments',
                overlaying='y',
                side='right'
            )
        )
        
        # Save visualization
        viz_path = self.results_dir / f'seasonal_patterns_{year}.html'
        fig.write_html(str(viz_path))
        print(f"Visualization saved to {viz_path}")
        
        return {
            'seasonal_data': seasonal_df,
            'visualization_path': str(viz_path)
        }
    
    def analyze_trade_corridors(self, year: str = '2020') -> Dict:
        """Analyze origin-destination pairs"""
        print(f"\nAnalyzing trade corridors for {year}...")
        
        # Get all CSV files for the specified year
        year_path = self.base_dir / year
        all_files = []
        for month_dir in year_path.glob("*/*/*.csv"):
            all_files.append(month_dir)
        
        # Process files
        corridor_data = []
        for file in tqdm(all_files, desc="Processing files"):
            df = self._process_file(file)
            if not df.empty:
                # Create origin and destination columns
                df['origin'] = df['USASTATE'].fillna('Unknown')
                df['destination'] = 'Unknown'
                mask_mex = df['COUNTRY'] == '1220'
                mask_can = df['COUNTRY'] == '2010'
                
                df.loc[mask_mex, 'destination'] = df.loc[mask_mex, 'MEXSTATE'].fillna('Unknown')
                df.loc[mask_can, 'destination'] = df.loc[mask_can, 'CANPROV'].fillna('Unknown')
                
                # Group by origin-destination pairs
                corridors = df.groupby(['origin', 'destination']).agg({
                    'VALUE': 'sum',
                    'SHIPWT': 'sum',
                    'FREIGHT_CHARGES': 'sum'
                }).reset_index()
                corridor_data.append(corridors)
        
        # Combine all corridor data
        combined_corridors = pd.concat(corridor_data, ignore_index=True)
        
        # Get top corridors
        top_corridors = combined_corridors.nlargest(20, 'VALUE')
        
        # Create visualization
        fig = px.bar(top_corridors,
                    x='origin',
                    y='VALUE',
                    color='destination',
                    title=f'Top 20 Trade Corridors by Value ({year})')
        
        fig.update_layout(
            xaxis_title='Origin (US State)',
            yaxis_title='Total Value',
            xaxis={'tickangle': 45}
        )
        
        # Save visualization
        viz_path = self.results_dir / f'trade_corridors_{year}.html'
        fig.write_html(str(viz_path))
        print(f"Visualization saved to {viz_path}")
        
        return {
            'corridor_data': top_corridors,
            'visualization_path': str(viz_path)
        }
    
    def analyze_commodity_distribution(self, year: str = '2020') -> Dict:
        """Analyze distribution of commodities"""
        print(f"\nAnalyzing commodity distribution for {year}...")
        
        # Get all CSV files for the specified year
        year_path = self.base_dir / year
        all_files = []
        for month_dir in year_path.glob("*/*/*.csv"):
            all_files.append(month_dir)
        
        # Process files
        commodity_data = []
        for file in tqdm(all_files, desc="Processing files"):
            df = self._process_file(file)
            if not df.empty:
                # Group by commodity code (DEPE)
                commodities = df.groupby('DEPE').agg({
                    'VALUE': 'sum',
                    'SHIPWT': 'sum'
                }).reset_index()
                commodity_data.append(commodities)
        
        # Combine all commodity data
        combined_commodities = pd.concat(commodity_data, ignore_index=True)
        
        # Get top commodities
        top_commodities = combined_commodities.nlargest(15, 'VALUE')
        
        # Create visualization
        fig = px.treemap(top_commodities,
                        path=['DEPE'],
                        values='VALUE',
                        title=f'Top 15 Commodities by Value ({year})')
        
        # Save visualization
        viz_path = self.results_dir / f'commodity_distribution_{year}.html'
        fig.write_html(str(viz_path))
        print(f"Visualization saved to {viz_path}")
        
        return {
            'commodity_data': top_commodities,
            'visualization_path': str(viz_path)
        }

def main():
    """Main function to run the analysis"""
    # Initialize analysis
    data_dir = Path(__file__).parent.parent / "data"
    analysis = DetailedPatternsAnalysis(data_dir)
    
    # Analyze patterns for 2020
    year = '2020'
    
    # Analyze seasonal patterns
    seasonal_results = analysis.analyze_seasonal_patterns(year)
    
    # Analyze trade corridors
    corridor_results = analysis.analyze_trade_corridors(year)
    
    # Analyze commodity distribution
    commodity_results = analysis.analyze_commodity_distribution(year)
    
    print("\nAnalysis complete! Results saved in:", analysis.results_dir)

if __name__ == "__main__":
    main()
