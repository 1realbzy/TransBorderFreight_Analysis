#!/usr/bin/env python3
"""
Test script to validate FreightDataProcessor functionality with path fix.
This script tests the core functionality without modifying the original.
"""

import sys
import os
from pathlib import Path

# Add the scripts directory to Python path
script_dir = Path(__file__).parent / 'scripts'
sys.path.insert(0, str(script_dir))

# Import the original module
from process_freight_data import FreightDataProcessor

class TestFreightDataProcessor(FreightDataProcessor):
    """Test version with fixed paths."""
    
    def __init__(self):
        # Use relative paths instead of hardcoded Windows path
        self.base_dir = Path("./data")
        self.output_dir = Path("./test_output")
        self.output_dir.mkdir(exist_ok=True)
        
        # Transport mode mapping
        self.mode_mapping = {
            1: 'RAIL',
            3: 'ROAD',
            4: 'AIR',
            5: 'VESSEL',
            6: 'MAIL',
            7: 'PIPELINE',
            8: 'OTHER',
            9: 'UNKNOWN'
        }
        
        # Define emission factors (kg CO2 per ton-mile)
        self.emission_factors = {
            'RAIL': 0.023,       # EPA estimates
            'ROAD': 0.161,       # EPA estimates
            'AIR': 1.527,        # ICAO estimates
            'VESSEL': 0.048,     # IMO estimates
            'PIPELINE': 0.015,   # Industry average
            'MAIL': 0.161,       # Assume similar to road
            'OTHER': 0.161,      # Conservative estimate
            'UNKNOWN': 0.161     # Conservative estimate
        }
        
        # Define safety risk factors (incidents per million ton-miles)
        self.safety_factors = {
            'RAIL': 0.15,
            'ROAD': 0.35,
            'AIR': 0.05,
            'VESSEL': 0.08,
            'PIPELINE': 0.02,
            'MAIL': 0.35,      # Assume similar to road
            'OTHER': 0.35,     # Conservative estimate
            'UNKNOWN': 0.35    # Conservative estimate
        }

        # Business Questions
        self.business_questions = [
            "What are the major trends in freight value and volume across transport modes?",
            "Which transport modes are most cost-efficient and environmentally sustainable?",
            "What are the seasonal patterns in freight movement and their implications?",
            "Which trade corridors show the highest growth potential?",
            "What are the safety risks and environmental impacts by transport mode?",
            "How has modal split evolved and what are the infrastructure implications?",
            "What strategic recommendations can improve freight efficiency and sustainability?"
        ]

def test_data_access():
    """Test data directory access and file enumeration."""
    print("=== Testing Data Access ===")
    
    processor = TestFreightDataProcessor()
    
    # Check data directory exists
    if not processor.base_dir.exists():
        print(f"❌ Data directory not found: {processor.base_dir}")
        return False
    
    print(f"✅ Data directory found: {processor.base_dir}")
    
    # Check yearly directories
    years_found = []
    for year in range(2020, 2025):
        year_dir = processor.base_dir / str(year)
        if year_dir.exists():
            years_found.append(year)
            
            # Check for extracted data
            extract_dir = year_dir / "extracted"
            if extract_dir.exists():
                csv_files = list(extract_dir.glob("**/*.csv"))
                print(f"✅ Year {year}: {len(csv_files)} CSV files found")
            else:
                print(f"⚠️  Year {year}: No extracted directory found")
    
    print(f"📊 Years available: {years_found}")
    return len(years_found) > 0

def test_single_year_processing():
    """Test processing a single year of data."""
    print("\n=== Testing Single Year Processing ===")
    
    processor = TestFreightDataProcessor()
    
    # Find a year with data
    test_year = None
    for year in range(2020, 2025):
        extract_dir = processor.base_dir / str(year) / "extracted"
        if extract_dir.exists() and list(extract_dir.glob("**/*.csv")):
            test_year = year
            break
    
    if not test_year:
        print("❌ No year with CSV data found")
        return False
    
    print(f"🔄 Testing with year {test_year}")
    
    try:
        # Process CSV files
        df = processor.process_csv_files(test_year)
        
        if df.empty:
            print("❌ No data processed")
            return False
        
        print(f"✅ Processed {len(df)} rows of data")
        print(f"📊 Columns: {list(df.columns)}")
        print(f"📊 Transport modes: {df['DISAGMOT'].unique() if 'DISAGMOT' in df.columns else 'N/A'}")
        
        # Test analysis
        if len(df) > 0:
            analysis = processor.analyze_data(df)
            if analysis:
                print(f"✅ Analysis completed with {len(analysis)} sections")
                print(f"📊 Analysis sections: {list(analysis.keys())}")
                return True
            else:
                print("❌ Analysis failed")
                return False
        
    except Exception as e:
        print(f"❌ Error during processing: {str(e)}")
        return False

def test_insight_generation():
    """Test insight generation with mock data."""
    print("\n=== Testing Insight Generation ===")
    
    processor = TestFreightDataProcessor()
    
    # Create mock analysis data
    mock_analysis = {
        2020: {
            'mode_analysis': {
                'modal_split': {'VESSEL': 65.0, 'RAIL': 7.5, 'ROAD': 4.3}
            },
            'efficiency': {
                'RAIL': {'mean': 100.0, 'median': 80.0},
                'ROAD': {'mean': 150.0, 'median': 120.0}
            },
            'environmental_impact': {
                'RAIL': {'emissions_per_ton': 0.023},
                'ROAD': {'emissions_per_ton': 0.161}
            },
            'seasonal_patterns': {
                1: {'RAIL': {'value': 1000, 'weight': 500}},
                2: {'RAIL': {'value': 1200, 'weight': 600}}
            },
            'trade_corridors': {
                'TX': {'RAIL': {'value': 5000, 'weight': 2500}}
            }
        }
    }
    
    try:
        insights = processor.generate_insights(mock_analysis)
        
        if not insights:
            print("❌ No insights generated")
            return False
        
        print(f"✅ Generated insights with {len(insights)} categories")
        print(f"📊 Categories: {list(insights.keys())}")
        
        # Check business question answers
        if 'business_question_answers' in insights:
            questions = insights['business_question_answers']
            print(f"✅ Answered {len(questions)} business questions")
            
            # Show sample answer
            if questions:
                first_question = list(questions.keys())[0]
                print(f"📝 Sample Q&A: {first_question[:50]}...")
                print(f"   Answer: {questions[first_question][0][:100]}...")
        
        return True
        
    except Exception as e:
        print(f"❌ Error during insight generation: {str(e)}")
        return False

def main():
    """Run all tests."""
    print("🚀 TransBorderFreight_Analysis - Functionality Test")
    print("=" * 60)
    
    tests = [
        ("Data Access", test_data_access),
        ("Single Year Processing", test_single_year_processing),
        ("Insight Generation", test_insight_generation)
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"❌ {test_name} failed with exception: {str(e)}")
            results.append((test_name, False))
    
    print("\n" + "=" * 60)
    print("🔍 Test Summary:")
    
    passed = 0
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"  {status} - {test_name}")
        if result:
            passed += 1
    
    print(f"\n📊 Results: {passed}/{len(results)} tests passed")
    
    if passed == len(results):
        print("🎉 All tests passed! The codebase is functional.")
    else:
        print("⚠️  Some tests failed. Check the output above for details.")
    
    return passed == len(results)

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)