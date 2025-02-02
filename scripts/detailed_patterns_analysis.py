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
                    df = pd.read_csv(file_path, dtype=str, encoding=encoding)
                    break
                except UnicodeDecodeError:
                    continue
            
            if df is None:
                raise ValueError("Could not read file with any encoding")
            
            # Clean the data
            # Convert numeric columns, replacing any invalid values with 0
            numeric_cols = ['VALUE', 'SHIPWT', 'FREIGHT_CHARGES']
            for col in numeric_cols:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col].str.replace(',', ''), errors='coerce').fillna(0)
            
            # Handle different file types (dot1, dot2, dot3)
            file_type = file_path.stem.split('_')[0]  # Get dot1, dot2, or dot3
            
            if file_type == 'dot3':
                # dot3 files don't have state/province info
                df['USASTATE'] = 'Aggregated'
                df['MEXSTATE'] = ''
                df['CANPROV'] = ''
            else:
                # Fill missing values for dot1 and dot2 files
                df['USASTATE'] = df['USASTATE'].fillna('Unknown')
                df['MEXSTATE'] = df['MEXSTATE'].fillna('')
                df['CANPROV'] = df['CANPROV'].fillna('')
            
            # Common missing value handling
            df['COUNTRY'] = df['COUNTRY'].fillna('0000')
            df['DEPE'] = df['DEPE'].fillna('Unknown')
            df['DISAGMOT'] = df['DISAGMOT'].fillna('Unknown')
            
            # Add file metadata
            df['source_file'] = file_path.name
            df['file_type'] = file_type
            
            return df
            
        except Exception as e:
            print(f"Error processing {file_path}: {str(e)}")
            return pd.DataFrame()
    
    def _find_csv_files(self, year: str) -> List[Path]:
        """Find all CSV files for a given year, handling different directory structures"""
        year_path = self.base_dir / year
        all_files = []
        
        # Search recursively for all CSV files
        for csv_file in year_path.rglob("*.csv"):
            # Skip YTD files as they are cumulative
            if "ytd" not in csv_file.name.lower():
                # Only process dot1 files for now to avoid double counting
                if csv_file.name.startswith("dot1"):
                    all_files.append(csv_file)
        
        return all_files
    
    def analyze_seasonal_patterns(self, year: str = '2020') -> Dict:
        """Analyze seasonal patterns within a year"""
        print(f"\nAnalyzing seasonal patterns for {year}...")
        
        # Get all CSV files for the specified year
        all_files = self._find_csv_files(year)
        
        if not all_files:
            raise ValueError(f"No data files found for year {year}")
        
        # Process files
        monthly_data = []
        for file in tqdm(all_files, desc="Processing files"):
            df = self._process_file(file)
            if not df.empty:
                monthly_stats = {
                    'month': df['MONTH'].iloc[0],
                    'month_name': calendar.month_name[int(df['MONTH'].iloc[0])],
                    'total_value': df['VALUE'].sum(),
                    'total_weight': df['SHIPWT'].sum(),
                    'num_shipments': len(df),
                    'avg_value_per_shipment': df['VALUE'].sum() / len(df) if len(df) > 0 else 0
                }
                monthly_data.append(monthly_stats)
        
        if not monthly_data:
            raise ValueError(f"No valid data processed for year {year}")
        
        # Convert to DataFrame and sort by month
        seasonal_df = pd.DataFrame(monthly_data)
        seasonal_df['month'] = pd.to_numeric(seasonal_df['month'])
        seasonal_df = seasonal_df.sort_values('month')
        
        # Create visualization
        fig = go.Figure()
        
        # Add value line
        fig.add_trace(go.Scatter(
            x=seasonal_df['month_name'],
            y=seasonal_df['total_value'],
            name='Total Value',
            line=dict(color='blue')
        ))
        
        # Add average value per shipment line
        fig.add_trace(go.Scatter(
            x=seasonal_df['month_name'],
            y=seasonal_df['avg_value_per_shipment'],
            name='Avg Value/Shipment',
            line=dict(color='green'),
            yaxis='y2'
        ))
        
        fig.update_layout(
            title=f'Seasonal Patterns in {year}',
            xaxis_title='Month',
            yaxis_title='Total Value',
            yaxis2=dict(
                title='Average Value per Shipment',
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
        all_files = self._find_csv_files(year)
        
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
        all_files = self._find_csv_files(year)
        
        if not all_files:
            raise ValueError(f"No data files found for year {year}")
        
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
        
        if not commodity_data:
            raise ValueError(f"No valid data processed for year {year}")
        
        # Combine all commodity data
        combined_commodities = pd.concat(commodity_data, ignore_index=True)
        
        # Aggregate by commodity code
        final_commodities = combined_commodities.groupby('DEPE').agg({
            'VALUE': 'sum',
            'SHIPWT': 'sum'
        }).reset_index()
        
        # Calculate percentages
        final_commodities['value_percentage'] = (final_commodities['VALUE'] / final_commodities['VALUE'].sum()) * 100
        final_commodities['weight_percentage'] = (final_commodities['SHIPWT'] / final_commodities['SHIPWT'].sum()) * 100
        
        # Get top commodities
        top_commodities = final_commodities.nlargest(15, 'VALUE')
        
        # Create visualization
        fig = go.Figure()
        
        # Add value percentage bars
        fig.add_trace(go.Bar(
            name='Value %',
            x=top_commodities['DEPE'],
            y=top_commodities['value_percentage'],
            marker_color='blue',
            opacity=0.7
        ))
        
        # Add weight percentage bars
        fig.add_trace(go.Bar(
            name='Weight %',
            x=top_commodities['DEPE'],
            y=top_commodities['weight_percentage'],
            marker_color='red',
            opacity=0.7
        ))
        
        fig.update_layout(
            title=f'Top 15 Commodities by Value and Weight ({year})',
            barmode='group',
            xaxis_title='Commodity Code',
            yaxis_title='Percentage (%)',
            xaxis={'tickangle': 45}
        )
        
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
