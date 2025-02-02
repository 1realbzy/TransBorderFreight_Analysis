"""
Freight Movement Patterns Analysis
--------------------------------
This module focuses on analyzing freight movement patterns across different transportation modes,
identifying trends in volume, routing, and modes of transportation across regions and time periods.
"""

import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
from typing import Dict, List
import glob
import os
from tqdm import tqdm

class FreightPatternsAnalysis:
    def __init__(self, base_dir: str):
        """
        Initialize the freight patterns analysis
        Args:
            base_dir: Base directory containing the data
        """
        self.base_dir = Path(base_dir)
        self.results_dir = self.base_dir.parent / "output" / "freight_patterns"
        self.results_dir.mkdir(exist_ok=True, parents=True)
        
        # Define mappings
        self.mode_mapping = {
            '1': 'Truck',
            '2': 'Rail',
            '3': 'Pipeline',
            '4': 'Air',
            '5': 'Vessel',
            '6': 'Mail',
            '7': 'Foreign Trade Zones',
            '8': 'Auto',
            '9': 'Other'
        }
        
        self.country_mapping = {
            '1220': 'Mexico',
            '2010': 'Canada'
        }
        
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
            # Replace any non-finite values with 0
            numeric_cols = ['VALUE', 'SHIPWT', 'FREIGHT_CHARGES']
            for col in numeric_cols:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
            
            # Add file metadata
            df['source_file'] = file_path.name
            return df
            
        except Exception as e:
            print(f"Error processing {file_path}: {str(e)}")
            return pd.DataFrame()
    
    def analyze_transport_modes(self, year: str = '2020') -> Dict:
        """
        Analyze distribution and trends of different transportation modes
        Args:
            year: Year to analyze (default: '2020')
        Returns:
            Dictionary containing analysis results
        """
        print(f"Analyzing transport modes for {year}...")
        
        # Get all CSV files for the specified year
        year_path = self.base_dir / year
        all_files = []
        for month_dir in year_path.glob("*/*/*.csv"):
            all_files.append(month_dir)
        
        # Process files
        dfs = []
        for file in tqdm(all_files, desc="Processing files"):
            df = self._process_file(file)
            if not df.empty:
                dfs.append(df)
        
        if not dfs:
            raise ValueError(f"No data found for year {year}")
            
        # Combine all data
        combined_df = pd.concat(dfs, ignore_index=True)
        
        # Calculate mode distribution
        mode_stats = combined_df.groupby('DISAGMOT').agg({
            'VALUE': 'sum',
            'SHIPWT': 'sum'
        })
        
        # Add mode names
        mode_stats.index = mode_stats.index.map(lambda x: self.mode_mapping.get(x, 'Unknown'))
        
        # Calculate percentages
        mode_stats['value_percentage'] = mode_stats['VALUE'] / mode_stats['VALUE'].sum() * 100
        mode_stats['weight_percentage'] = mode_stats['SHIPWT'] / mode_stats['SHIPWT'].sum() * 100
        
        # Create visualization
        fig = go.Figure()
        
        # Add value percentage bars
        fig.add_trace(go.Bar(
            name='Value %',
            x=mode_stats.index,
            y=mode_stats['value_percentage'],
            marker_color='blue',
            opacity=0.7
        ))
        
        # Add weight percentage bars
        fig.add_trace(go.Bar(
            name='Weight %',
            x=mode_stats.index,
            y=mode_stats['weight_percentage'],
            marker_color='red',
            opacity=0.7
        ))
        
        fig.update_layout(
            title=f'Transportation Mode Distribution by Value and Weight ({year})',
            barmode='group',
            xaxis_title='Transportation Mode',
            yaxis_title='Percentage (%)',
            xaxis={'tickangle': 45}
        )
        
        # Save visualization
        viz_path = self.results_dir / f'transport_modes_{year}.html'
        fig.write_html(str(viz_path))
        print(f"Visualization saved to {viz_path}")
        
        return {
            'mode_stats': mode_stats,
            'visualization_path': str(viz_path)
        }
    
    def analyze_regional_patterns(self, year: str = '2020') -> Dict:
        """
        Analyze regional patterns in freight movement
        Args:
            year: Year to analyze (default: '2020')
        Returns:
            Dictionary containing analysis results
        """
        print(f"Analyzing regional patterns for {year}...")
        
        # Get all CSV files for the specified year
        year_path = self.base_dir / year
        all_files = []
        for month_dir in year_path.glob("*/*/*.csv"):
            all_files.append(month_dir)
        
        # Process files
        dfs = []
        for file in tqdm(all_files, desc="Processing files"):
            df = self._process_file(file)
            if not df.empty:
                dfs.append(df)
        
        if not dfs:
            raise ValueError(f"No data found for year {year}")
            
        # Combine all data
        combined_df = pd.concat(dfs, ignore_index=True)
        
        # Analyze patterns by country and state/province
        regional_stats = {}
        
        # US States analysis
        us_stats = combined_df.groupby('USASTATE').agg({
            'VALUE': 'sum',
            'SHIPWT': 'sum'
        }).nlargest(10, 'VALUE')
        
        # Mexico States analysis
        mex_stats = combined_df[combined_df['COUNTRY'] == '1220'].groupby('MEXSTATE').agg({
            'VALUE': 'sum',
            'SHIPWT': 'sum'
        }).nlargest(10, 'VALUE')
        
        # Canada Provinces analysis
        can_stats = combined_df[combined_df['COUNTRY'] == '2010'].groupby('CANPROV').agg({
            'VALUE': 'sum',
            'SHIPWT': 'sum'
        }).nlargest(10, 'VALUE')
        
        regional_stats = {
            'us_states': us_stats,
            'mexico_states': mex_stats,
            'canada_provinces': can_stats
        }
        
        # Create visualizations
        viz_paths = {}
        for region_type, data in regional_stats.items():
            fig = px.bar(data,
                        x=data.index,
                        y='VALUE',
                        title=f'Top 10 {region_type.replace("_", " ").title()} by Trade Value ({year})')
            
            fig.update_layout(
                xaxis_title='State/Province',
                yaxis_title='Total Value',
                xaxis={'tickangle': 45}
            )
            
            viz_path = self.results_dir / f'{region_type}_{year}.html'
            fig.write_html(str(viz_path))
            viz_paths[region_type] = str(viz_path)
            print(f"Visualization saved to {viz_path}")
        
        return {
            'regional_stats': regional_stats,
            'visualization_paths': viz_paths
        }

def main():
    """Main function to run the analysis"""
    # Initialize analysis
    data_dir = Path(__file__).parent.parent / "data"
    analysis = FreightPatternsAnalysis(data_dir)
    
    # Analyze transport modes for 2020
    print("\nAnalyzing transport modes...")
    mode_results = analysis.analyze_transport_modes(year='2020')
    
    # Print key findings
    print("\nKey Findings - Transport Modes:")
    mode_stats = mode_results['mode_stats']
    for mode in mode_stats.index:
        print(f"   - {mode}: {mode_stats.loc[mode, 'value_percentage']:.1f}% by value, "
              f"{mode_stats.loc[mode, 'weight_percentage']:.1f}% by weight")
    
    # Analyze regional patterns
    print("\nAnalyzing regional patterns...")
    regional_results = analysis.analyze_regional_patterns(year='2020')
    
    print("\nAnalysis complete! Results saved in:", analysis.results_dir)

if __name__ == "__main__":
    main()
