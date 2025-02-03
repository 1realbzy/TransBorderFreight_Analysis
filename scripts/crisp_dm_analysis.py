"""
CRISP-DM Framework Implementation for TransBorder Freight Analysis
This module implements the Cross-Industry Standard Process for Data Mining methodology
"""

import pandas as pd
import numpy as np
from pathlib import Path
import logging
from typing import Dict, List, Tuple
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import json
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import r2_score, mean_squared_error

class CRISPDMAnalysis:
    def __init__(self, data_dir: Path):
        self.data_dir = data_dir
        self.output_dir = data_dir.parent / "output" / "crisp_dm_analysis"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.logger = logging.getLogger(__name__)
        
        # Define business questions
        self.business_questions = {
            "Q1": "What are the major patterns and trends in freight movement across different transport modes from 2020-2024?",
            "Q2": "How has operational efficiency evolved, and what are the key factors affecting it?",
            "Q3": "What is the environmental impact of different transport modes, and how can it be optimized?",
            "Q4": "What are the main safety concerns and risk factors in freight transportation?",
            "Q5": "How have economic disruptions affected freight patterns and what are the recovery patterns?",
            "Q6": "Which routes and transport modes show the highest efficiency and lowest environmental impact?",
            "Q7": "What are the seasonal patterns and their implications for freight planning?"
        }

    def business_understanding(self) -> Dict:
        """
        Phase 1: Business Understanding
        - Define business objectives
        - Assess situation
        - Determine data mining goals
        - Produce project plan
        """
        try:
            business_context = {
                "objectives": {
                    "primary": "Optimize freight transportation operations while minimizing environmental impact",
                    "secondary": [
                        "Identify operational inefficiencies",
                        "Assess environmental impact",
                        "Evaluate safety metrics",
                        "Analyze economic disruptions",
                        "Provide data-driven recommendations"
                    ]
                },
                "success_criteria": {
                    "business": [
                        "Identification of cost-saving opportunities",
                        "Reduction in environmental impact",
                        "Improvement in safety metrics",
                        "Enhanced operational efficiency"
                    ],
                    "technical": [
                        "Accurate prediction of freight patterns",
                        "Reliable identification of inefficiencies",
                        "Comprehensive risk assessment",
                        "Statistically significant findings"
                    ]
                },
                "analytical_questions": self.business_questions
            }
            
            # Save business understanding documentation
            with open(self.output_dir / "business_understanding.json", 'w') as f:
                json.dump(business_context, f, indent=2)
            
            return business_context
            
        except Exception as e:
            self.logger.error(f"Error in business understanding phase: {str(e)}")
            raise

    def data_understanding(self, df: pd.DataFrame) -> Dict:
        """
        Phase 2: Data Understanding
        - Collect initial data
        - Describe data
        - Explore data
        - Verify data quality
        """
        try:
            data_profile = {
                "basic_stats": self._generate_basic_stats(df),
                "data_quality": self._assess_data_quality(df),
                "feature_analysis": self._analyze_features(df),
                "temporal_patterns": self._analyze_temporal_patterns(df)
            }
            
            # Save data understanding documentation
            with open(self.output_dir / "data_understanding.json", 'w') as f:
                json.dump(data_profile, f, indent=2)
            
            return data_profile
            
        except Exception as e:
            self.logger.error(f"Error in data understanding phase: {str(e)}")
            raise

    def data_preparation(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Phase 3: Data Preparation
        - Select data
        - Clean data
        - Construct data
        - Integrate data
        - Format data
        """
        try:
            # Document preparation steps
            prep_steps = {
                "cleaning": [],
                "feature_engineering": [],
                "transformations": []
            }
            
            # Handle missing values
            if df.isnull().sum().any():
                df = self._handle_missing_values(df)
                prep_steps["cleaning"].append("Handled missing values")
            
            # Handle outliers
            df = self._handle_outliers(df)
            prep_steps["cleaning"].append("Handled outliers")
            
            # Feature engineering
            df = self._engineer_features(df)
            prep_steps["feature_engineering"].extend([
                "Added efficiency metrics",
                "Created temporal features",
                "Calculated environmental indicators"
            ])
            
            # Save preparation documentation
            with open(self.output_dir / "data_preparation.json", 'w') as f:
                json.dump(prep_steps, f, indent=2)
            
            return df
            
        except Exception as e:
            self.logger.error(f"Error in data preparation phase: {str(e)}")
            raise

    def modeling(self, df: pd.DataFrame) -> Dict:
        """
        Phase 4: Modeling
        - Select modeling techniques
        - Generate test design
        - Build models
        - Assess models
        """
        try:
            modeling_results = {}
            
            # Time series forecasting for freight patterns
            modeling_results['freight_patterns'] = self._model_freight_patterns(df)
            
            # Efficiency prediction model
            modeling_results['efficiency_prediction'] = self._model_efficiency(df)
            
            # Environmental impact modeling
            modeling_results['environmental_impact'] = self._model_environmental_impact(df)
            
            # Risk assessment model
            modeling_results['risk_assessment'] = self._model_risk_assessment(df)
            
            # Save modeling documentation
            with open(self.output_dir / "modeling_results.json", 'w') as f:
                json.dump(modeling_results, f, indent=2)
            
            return modeling_results
            
        except Exception as e:
            self.logger.error(f"Error in modeling phase: {str(e)}")
            raise

    def evaluation(self, df: pd.DataFrame, modeling_results: Dict) -> Dict:
        """
        Phase 5: Evaluation
        - Evaluate results
        - Review process
        - Determine next steps
        """
        try:
            evaluation_results = {
                "model_performance": self._evaluate_model_performance(modeling_results),
                "business_objectives": self._evaluate_business_objectives(df, modeling_results),
                "insights": self._generate_key_insights(df, modeling_results),
                "recommendations": self._generate_recommendations(df, modeling_results)
            }
            
            # Save evaluation documentation
            with open(self.output_dir / "evaluation_results.json", 'w') as f:
                json.dump(evaluation_results, f, indent=2)
            
            return evaluation_results
            
        except Exception as e:
            self.logger.error(f"Error in evaluation phase: {str(e)}")
            raise

    def deployment(self, evaluation_results: Dict) -> Dict:
        """
        Phase 6: Deployment
        - Plan deployment
        - Plan monitoring and maintenance
        - Produce final report
        - Review project
        """
        try:
            deployment_plan = {
                "implementation_steps": self._create_implementation_plan(evaluation_results),
                "monitoring_plan": self._create_monitoring_plan(evaluation_results),
                "maintenance_schedule": self._create_maintenance_schedule(),
                "final_report": self._generate_final_report(evaluation_results)
            }
            
            # Save deployment documentation
            with open(self.output_dir / "deployment_plan.json", 'w') as f:
                json.dump(deployment_plan, f, indent=2)
            
            return deployment_plan
            
        except Exception as e:
            self.logger.error(f"Error in deployment phase: {str(e)}")
            raise

    # Helper methods for data understanding
    def _generate_basic_stats(self, df: pd.DataFrame) -> Dict:
        """Generate basic statistical analysis of the data"""
        stats = {
            "record_count": len(df),
            "feature_count": len(df.columns),
            "numeric_features": df.select_dtypes(include=[np.number]).columns.tolist(),
            "categorical_features": df.select_dtypes(include=['object']).columns.tolist(),
            "time_range": {
                "start": df['Date'].min().strftime('%Y-%m-%d'),
                "end": df['Date'].max().strftime('%Y-%m-%d')
            }
        }
        return stats

    def _assess_data_quality(self, df: pd.DataFrame) -> Dict:
        """Assess the quality of the data"""
        quality = {
            "missing_values": df.isnull().sum().to_dict(),
            "duplicates": df.duplicated().sum(),
            "outliers": self._detect_outliers(df)
        }
        return quality

    def _analyze_features(self, df: pd.DataFrame) -> Dict:
        """Analyze individual features"""
        feature_analysis = {}
        for col in df.columns:
            if df[col].dtype in [np.number]:
                feature_analysis[col] = {
                    "mean": df[col].mean(),
                    "median": df[col].median(),
                    "std": df[col].std(),
                    "min": df[col].min(),
                    "max": df[col].max()
                }
            else:
                feature_analysis[col] = {
                    "unique_values": df[col].nunique(),
                    "top_values": df[col].value_counts().head().to_dict()
                }
        return feature_analysis

    def _analyze_temporal_patterns(self, df: pd.DataFrame) -> Dict:
        """Analyze temporal patterns in the data"""
        temporal = {
            "yearly_patterns": df.groupby(df['Date'].dt.year)['VALUE'].sum().to_dict(),
            "monthly_patterns": df.groupby(df['Date'].dt.month)['VALUE'].mean().to_dict(),
            "seasonal_patterns": df.groupby(df['Date'].dt.quarter)['VALUE'].mean().to_dict()
        }
        return temporal

    # Helper methods for data preparation
    def _handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """Handle missing values in the dataset"""
        # Numeric columns: interpolate
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        df[numeric_cols] = df[numeric_cols].interpolate(method='time')
        
        # Categorical columns: forward fill
        categorical_cols = df.select_dtypes(include=['object']).columns
        df[categorical_cols] = df[categorical_cols].fillna(method='ffill')
        
        return df

    def _handle_outliers(self, df: pd.DataFrame) -> pd.DataFrame:
        """Handle outliers in the dataset"""
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            df[col] = df[col].clip(lower_bound, upper_bound)
        return df

    def _engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Engineer new features"""
        # Efficiency metrics
        df['ValueDensity'] = df['VALUE'] / df['SHIPWT']
        
        # Temporal features
        df['Year'] = df['Date'].dt.year
        df['Month'] = df['Date'].dt.month
        df['Quarter'] = df['Date'].dt.quarter
        df['Season'] = df['Month'].map({
            12:'Winter', 1:'Winter', 2:'Winter',
            3:'Spring', 4:'Spring', 5:'Spring',
            6:'Summer', 7:'Summer', 8:'Summer',
            9:'Fall', 10:'Fall', 11:'Fall'
        })
        
        return df

    # Helper methods for modeling
    def _model_freight_patterns(self, df: pd.DataFrame) -> Dict:
        """Model freight patterns using time series analysis and forecasting"""
        try:
            results = {}
            
            # Analyze patterns by transport mode
            for mode in df['DISAGMOT'].unique():
                mode_data = df[df['DISAGMOT'] == mode].copy()
                mode_data = mode_data.set_index('Date')
                mode_data = mode_data.sort_index()
                
                # Decompose time series
                decomposition = seasonal_decompose(mode_data['VALUE'], period=12)
                
                # Forecast future values
                model = SARIMAX(mode_data['VALUE'], 
                              order=(1, 1, 1), 
                              seasonal_order=(1, 1, 1, 12))
                results_sarima = model.fit()
                forecast = results_sarima.forecast(steps=12)
                
                results[mode] = {
                    'trend': decomposition.trend.dropna().tolist(),
                    'seasonal': decomposition.seasonal.dropna().tolist(),
                    'residual': decomposition.resid.dropna().tolist(),
                    'forecast': forecast.tolist(),
                    'forecast_conf_int': results_sarima.get_forecast(steps=12).conf_int().values.tolist()
                }
            
            return results
            
        except Exception as e:
            self.logger.error(f"Error in freight pattern modeling: {str(e)}")
            raise

    def _model_efficiency(self, df: pd.DataFrame) -> Dict:
        """Model operational efficiency using machine learning"""
        try:
            results = {}
            
            # Prepare features for efficiency modeling
            features = ['SHIPWT', 'Year', 'Month', 'ValueDensity']
            target = 'VALUE'
            
            # Split data
            X = df[features]
            y = df[target]
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            
            # Scale features
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            
            # Train model
            model = RandomForestRegressor(n_estimators=100, random_state=42)
            model.fit(X_train_scaled, y_train)
            
            # Get feature importance
            feature_importance = dict(zip(features, model.feature_importances_))
            
            # Make predictions
            y_pred = model.predict(X_test_scaled)
            
            results = {
                'model_performance': {
                    'r2_score': r2_score(y_test, y_pred),
                    'mse': mean_squared_error(y_test, y_pred)
                },
                'feature_importance': feature_importance,
                'efficiency_scores': {
                    'mean_efficiency': df['ValueDensity'].mean(),
                    'efficiency_trend': df.groupby('Year')['ValueDensity'].mean().to_dict()
                }
            }
            
            return results
            
        except Exception as e:
            self.logger.error(f"Error in efficiency modeling: {str(e)}")
            raise

    def _model_environmental_impact(self, df: pd.DataFrame) -> Dict:
        """Model environmental impact using statistical analysis"""
        try:
            results = {}
            
            # Calculate emissions by transport mode
            emissions_factors = {
                'Truck': 0.2,  # kg CO2 per ton-mile
                'Rail': 0.05,
                'Water': 0.03,
                'Air': 0.8
            }
            
            df['EmissionsEstimate'] = df.apply(
                lambda row: row['SHIPWT'] * emissions_factors.get(row['DISAGMOT'], 0.2),
                axis=1
            )
            
            # Analyze environmental impact by mode
            for mode in df['DISAGMOT'].unique():
                mode_data = df[df['DISAGMOT'] == mode]
                
                results[mode] = {
                    'total_emissions': mode_data['EmissionsEstimate'].sum(),
                    'emissions_per_value': (mode_data['EmissionsEstimate'].sum() / 
                                         mode_data['VALUE'].sum()),
                    'emissions_trend': mode_data.groupby('Year')['EmissionsEstimate'].mean().to_dict(),
                    'efficiency_score': (mode_data['VALUE'].sum() / 
                                      mode_data['EmissionsEstimate'].sum())
                }
            
            # Calculate overall environmental metrics
            results['overall'] = {
                'total_emissions': df['EmissionsEstimate'].sum(),
                'emissions_intensity': df['EmissionsEstimate'].sum() / df['VALUE'].sum(),
                'yearly_trend': df.groupby('Year')['EmissionsEstimate'].sum().to_dict()
            }
            
            return results
            
        except Exception as e:
            self.logger.error(f"Error in environmental impact modeling: {str(e)}")
            raise

    def _model_risk_assessment(self, df: pd.DataFrame) -> Dict:
        """Model risk assessment using statistical analysis and machine learning"""
        try:
            results = {}
            
            # Calculate value at risk by mode and route
            df['ValueAtRisk'] = df['VALUE'] * df['ValueDensity']
            
            # Identify high-risk routes
            route_risk = df.groupby(['USASTATE', 'DISAGMOT']).agg({
                'ValueAtRisk': ['mean', 'std'],
                'VALUE': 'sum'
            }).reset_index()
            
            # Calculate risk scores
            route_risk['RiskScore'] = (route_risk['ValueAtRisk']['std'] / 
                                     route_risk['ValueAtRisk']['mean'])
            
            # Identify risk factors
            risk_factors = {
                'high_value_routes': route_risk.nlargest(10, ('VALUE', 'sum')).to_dict('records'),
                'high_risk_routes': route_risk.nlargest(10, 'RiskScore').to_dict('records'),
                'risk_by_mode': df.groupby('DISAGMOT')['ValueAtRisk'].agg([
                    'mean', 'std', 'min', 'max'
                ]).to_dict('index')
            }
            
            # Seasonal risk analysis
            seasonal_risk = df.groupby(['Season', 'DISAGMOT'])['ValueAtRisk'].agg([
                'mean', 'std'
            ]).reset_index()
            
            results = {
                'risk_factors': risk_factors,
                'seasonal_risk': seasonal_risk.to_dict('records'),
                'risk_metrics': {
                    'total_value_at_risk': df['ValueAtRisk'].sum(),
                    'risk_concentration': df.groupby('DISAGMOT')['ValueAtRisk'].sum().to_dict()
                }
            }
            
            return results
            
        except Exception as e:
            self.logger.error(f"Error in risk assessment modeling: {str(e)}")
            raise

    def _evaluate_model_performance(self, modeling_results: Dict) -> Dict:
        """Evaluate performance of all models"""
        try:
            evaluation = {
                'freight_patterns': {
                    'forecast_accuracy': self._evaluate_forecast_accuracy(
                        modeling_results['freight_patterns']
                    ),
                    'pattern_strength': self._evaluate_pattern_strength(
                        modeling_results['freight_patterns']
                    )
                },
                'efficiency_model': {
                    'model_metrics': modeling_results['efficiency_prediction']['model_performance'],
                    'feature_importance': modeling_results['efficiency_prediction']['feature_importance']
                },
                'environmental_impact': {
                    'impact_assessment': self._evaluate_environmental_impact(
                        modeling_results['environmental_impact']
                    )
                },
                'risk_assessment': {
                    'risk_metrics': self._evaluate_risk_metrics(
                        modeling_results['risk_assessment']
                    )
                }
            }
            
            return evaluation
            
        except Exception as e:
            self.logger.error(f"Error in model performance evaluation: {str(e)}")
            raise

    def _evaluate_business_objectives(self, df: pd.DataFrame, modeling_results: Dict) -> Dict:
        """Evaluate achievement of business objectives"""
        try:
            evaluation = {
                'operational_efficiency': {
                    'improvement_opportunities': self._identify_efficiency_improvements(
                        modeling_results['efficiency_prediction']
                    ),
                    'cost_saving_potential': self._calculate_cost_savings(
                        modeling_results['efficiency_prediction']
                    )
                },
                'environmental_impact': {
                    'emission_reduction_potential': self._calculate_emission_reduction(
                        modeling_results['environmental_impact']
                    ),
                    'sustainability_metrics': self._evaluate_sustainability(
                        modeling_results['environmental_impact']
                    )
                },
                'risk_mitigation': {
                    'high_risk_areas': self._identify_risk_areas(
                        modeling_results['risk_assessment']
                    ),
                    'mitigation_strategies': self._develop_risk_strategies(
                        modeling_results['risk_assessment']
                    )
                },
                'economic_performance': {
                    'value_optimization': self._analyze_value_optimization(
                        modeling_results['freight_patterns']
                    ),
                    'growth_opportunities': self._identify_growth_areas(
                        modeling_results['freight_patterns']
                    )
                }
            }
            
            return evaluation
            
        except Exception as e:
            self.logger.error(f"Error in business objective evaluation: {str(e)}")
            raise

    def _generate_key_insights(self, df: pd.DataFrame, modeling_results: Dict) -> List[Dict]:
        """Generate key insights from the analysis"""
        try:
            insights = []
            
            # Operational insights
            insights.extend(self._generate_operational_insights(
                modeling_results['efficiency_prediction']
            ))
            
            # Environmental insights
            insights.extend(self._generate_environmental_insights(
                modeling_results['environmental_impact']
            ))
            
            # Risk insights
            insights.extend(self._generate_risk_insights(
                modeling_results['risk_assessment']
            ))
            
            # Economic insights
            insights.extend(self._generate_economic_insights(
                modeling_results['freight_patterns']
            ))
            
            return insights
            
        except Exception as e:
            self.logger.error(f"Error generating key insights: {str(e)}")
            raise

    def _generate_recommendations(self, df: pd.DataFrame, modeling_results: Dict) -> List[Dict]:
        """Generate actionable recommendations"""
        try:
            recommendations = []
            
            # Operational recommendations
            recommendations.extend(self._generate_operational_recommendations(
                modeling_results['efficiency_prediction']
            ))
            
            # Environmental recommendations
            recommendations.extend(self._generate_environmental_recommendations(
                modeling_results['environmental_impact']
            ))
            
            # Risk recommendations
            recommendations.extend(self._generate_risk_recommendations(
                modeling_results['risk_assessment']
            ))
            
            # Economic recommendations
            recommendations.extend(self._generate_economic_recommendations(
                modeling_results['freight_patterns']
            ))
            
            return recommendations
            
        except Exception as e:
            self.logger.error(f"Error generating recommendations: {str(e)}")
            raise
