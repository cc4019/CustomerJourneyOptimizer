import pandas as pd
import numpy as np
from typing import List, Dict, Any
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

class InterventionToHVAAnalyzer:
    def __init__(self, intervention_analyzer: Any, hva_tracker: Any):
        self.intervention_analyzer = intervention_analyzer
        self.hva_tracker = hva_tracker
        self.model = None

    def prepare_data(self, intervention_id: str, hva_id: str, 
                     feature_columns: List[str]) -> pd.DataFrame:
        """
        Prepare data for analyzing the impact of an intervention on an HVA.

        :param intervention_id: ID of the intervention to analyze
        :param hva_id: ID of the HVA to analyze
        :param feature_columns: List of feature column names to include in the analysis
        :return: DataFrame with prepared data
        """
        # Get intervention data
        intervention_data = self.intervention_analyzer.get_intervention_results(intervention_id)
        
        # Get HVA data
        hva_data = self.hva_tracker.get_hva_records(hva_id)
        
        # Merge intervention and HVA data
        merged_data = pd.merge(intervention_data, hva_data, on='customer_id', how='left')
        
        # Create target variable (whether HVA occurred after intervention)
        merged_data['hva_occurred'] = (merged_data['hva_timestamp'] > merged_data['intervention_timestamp']).astype(int)
        
        # Select features and target
        X = merged_data[feature_columns]
        y = merged_data['hva_occurred']
        
        return X, y

    def train_model(self, X: pd.DataFrame, y: pd.Series):
        """
        Train a model to predict HVA occurrence based on intervention and customer features.

        :param X: Feature DataFrame
        :param y: Target Series
        """
        # Split data into train and test sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Train model
        self.model = RandomForestClassifier(n_estimators=100, random_state=42)
        self.model.fit(X_train_scaled, y_train)
        
        # Evaluate model
        y_pred = self.model.predict(X_test_scaled)
        print(classification_report(y_test, y_pred))

    def analyze_intervention_impact(self, intervention_id: str, hva_id: str, 
                                    feature_columns: List[str]) -> Dict[str, Any]:
        """
        Analyze the impact of an intervention on an HVA.

        :param intervention_id: ID of the intervention to analyze
        :param hva_id: ID of the HVA to analyze
        :param feature_columns: List of feature column names to include in the analysis
        :return: Dictionary containing impact analysis results
        """
        # Prepare data
        X, y = self.prepare_data(intervention_id, hva_id, feature_columns)
        
        # Train model
        self.train_model(X, y)
        
        # Get feature importances
        feature_importances = pd.DataFrame({
            'feature': feature_columns,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        # Calculate overall impact
        overall_impact = y.mean()
        
        return {
            'intervention_id': intervention_id,
            'hva_id': hva_id,
            'overall_impact': overall_impact,
            'feature_importances': feature_importances,
            'model': self.model
        }

    def predict_hva_occurrence(self, customer_features: pd.DataFrame) -> np.ndarray:
        """
        Predict HVA occurrence probability for given customer features.

        :param customer_features: DataFrame of customer features
        :return: Array of HVA occurrence probabilities
        """
        if self.model is None:
            raise ValueError("Model has not been trained. Call analyze_intervention_impact first.")
        
        return self.model.predict_proba(customer_features)[:, 1]
