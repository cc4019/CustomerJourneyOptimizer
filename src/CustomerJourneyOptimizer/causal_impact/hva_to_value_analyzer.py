import pandas as pd
import numpy as np
from typing import List, Dict, Any
from statsmodels.tsa.statespace.structural import UnobservedComponents
from statsmodels.tsa.statespace.sarimax import SARIMAX

class HVAToValueAnalyzer:
    def __init__(self, hva_tracker: Any, value_data: pd.DataFrame):
        self.hva_tracker = hva_tracker
        self.value_data = value_data

    def analyze_hva_impact(self, hva_id: str, target_value: str, 
                           start_date: pd.Timestamp, end_date: pd.Timestamp) -> Dict[str, Any]:
        """
        Analyze the causal impact of an HVA on a target value.

        :param hva_id: ID of the HVA to analyze
        :param target_value: Name of the target value column in value_data
        :param start_date: Start date of the analysis period
        :param end_date: End date of the analysis period
        :return: Dictionary containing impact analysis results
        """
        # Get HVA timeline
        hva_timeline = self.hva_tracker.get_hva_timeline(hva_id, start_date, end_date)
        
        # Merge HVA timeline with value data
        merged_data = pd.merge(hva_timeline, self.value_data, left_on='date', right_index=True, how='left')
        
        # Fit structural time series model
        model = UnobservedComponents(merged_data[target_value], level='local linear trend', 
                                     seasonal=12, exog=merged_data['count'])
        results = model.fit()
        
        # Calculate impact
        impact = results.params['count'] * merged_data['count'].mean()
        
        return {
            'hva_id': hva_id,
            'target_value': target_value,
            'estimated_impact': impact,
            'p_value': results.pvalues['count'],
            'model_summary': results.summary()
        }

    def forecast_value(self, hva_id: str, target_value: str, 
                       forecast_horizon: int) -> pd.DataFrame:
        """
        Forecast the target value based on predicted HVA occurrences.

        :param hva_id: ID of the HVA to use for forecasting
        :param target_value: Name of the target value column in value_data
        :param forecast_horizon: Number of periods to forecast
        :return: DataFrame with forecasted values
        """
        # Get historical HVA data
        hva_data = self.hva_tracker.get_hva_timeline(hva_id, self.value_data.index.min(), self.value_data.index.max())
        
        # Merge HVA data with value data
        merged_data = pd.merge(hva_data, self.value_data, left_on='date', right_index=True, how='left')
        
        # Fit SARIMAX model
        model = SARIMAX(merged_data[target_value], exog=merged_data['count'], 
                        order=(1, 1, 1), seasonal_order=(1, 1, 1, 12))
        results = model.fit()
        
        # Forecast HVA occurrences (simple moving average forecast for demonstration)
        hva_forecast = pd.DataFrame({
            'count': [merged_data['count'].rolling(window=30).mean().iloc[-1]] * forecast_horizon
        }, index=pd.date_range(start=merged_data.index[-1] + pd.Timedelta(days=1), periods=forecast_horizon))
        
        # Generate value forecast
        forecast = results.get_forecast(steps=forecast_horizon, exog=hva_forecast)
        
        return pd.DataFrame({
            'forecasted_value': forecast.predicted_mean,
            'lower_ci': forecast.conf_int()['lower ' + target_value],
            'upper_ci': forecast.conf_int()['upper ' + target_value]
        })

    def compare_hva_impacts(self, hva_ids: List[str], target_value: str, 
                            start_date: pd.Timestamp, end_date: pd.Timestamp) -> pd.DataFrame:
        """
        Compare the impacts of multiple HVAs on a target value.

        :param hva_ids: List of HVA IDs to compare
        :param target_value: Name of the target value column in value_data
        :param start_date: Start date of the analysis period
        :param end_date: End date of the analysis period
        :return: DataFrame with comparison results
        """
        results = []
        for hva_id in hva_ids:
            impact = self.analyze_hva_impact(hva_id, target_value, start_date, end_date)
            results.append({
                'hva_id': hva_id,
                'estimated_impact': impact['estimated_impact'],
                'p_value': impact['p_value']
            })
        
        return pd.DataFrame(results).sort_values('estimated_impact', ascending=False)