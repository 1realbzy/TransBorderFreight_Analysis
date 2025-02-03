"""
Configuration settings for the TransBorder Freight Analysis
"""

# Analysis years
ANALYSIS_YEARS = ['2020', '2021', '2022', '2023', '2024']

# Environmental impact thresholds (based on industry standards)
EMISSIONS_THRESHOLDS = {
    'truck': 161.8,  # g CO2/ton-km
    'rail': 30.9,    # g CO2/ton-km
    'air': 673.0     # g CO2/ton-km
}

# Efficiency metrics
EFFICIENCY_METRICS = {
    'value_density_threshold': 1000,  # $/kg
    'delay_threshold': 48,           # hours
    'utilization_min': 0.75         # 75% minimum utilization
}

# Safety thresholds
SAFETY_THRESHOLDS = {
    'high_risk_value': 1000000,     # High-value shipments
    'weight_limit': 80000,          # lbs (standard US truck weight limit)
    'hazmat_codes': ['2716', '2806', '2849']  # Example hazardous material codes
}

# Economic indicators
ECONOMIC_INDICATORS = {
    'gdp_impact_threshold': 0.01,    # 1% of GDP
    'trade_balance_threshold': 0.1,  # 10% change
    'seasonal_variation_threshold': 0.25  # 25% variation
}

# Transport modes mapping
TRANSPORT_MODES = {
    '1': 'Truck',
    '2': 'Rail',
    '3': 'Pipeline',
    '4': 'Air',
    '5': 'Vessel',
    '6': 'Mail',
    '7': 'Other'
}

# Commodity categories of special interest
CRITICAL_COMMODITIES = {
    'MEDICAL': ['3002', '3003', '3004'],  # Medical supplies
    'TECH': ['8471', '8542', '8517'],     # Technology products
    'FOOD': ['0901', '1001', '1006'],     # Essential food items
    'ENERGY': ['2709', '2711', '2716']    # Energy products
}

# Geographic regions for corridor analysis
KEY_REGIONS = {
    'PORTS': ['NYNJ', 'LALB', 'SEATAC'],  # Major ports
    'BORDERS': ['DET', 'BUFF', 'LAREDO'],  # Border crossings
    'HUBS': ['CHI', 'ATL', 'DAL']         # Distribution hubs
}

# Time periods for temporal analysis
TIME_PERIODS = {
    'PRE_COVID': '2019-Q4',
    'COVID_PEAK': '2020-Q2',
    'RECOVERY': '2021-Q1',
    'POST_COVID': '2022-Q1',
    'CURRENT': '2024-Q4'
}
