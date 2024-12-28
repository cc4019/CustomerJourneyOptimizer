import pandas as pd
import numpy as np
from typing import Dict, List
from .intervention_catalog import InterventionCatalog

class InterventionAnalyzer:
    def __init__(self, intervention_catalog: InterventionCatalog):
        self.intervention_catalog = intervention_catalog
        self.intervention_results: Dict[str, Dict] = {}

    def record_intervention_result(self, intervention_id: str, customer_id: str, timestamp: str, outcome: str):
        """
        Record the result of an intervention for a specific customer.
        
        :param intervention_id: Unique identifier for the intervention
        :param customer_id: Unique identifier for the customer
        :param timestamp: Time when the intervention was applied
        :param outcome: Outcome of the intervention (e.g., 'success', 'failure')
        """
        if intervention_id not in self.intervention_results:
            self.intervention_results[intervention_id] = []
        
        self.intervention_results[intervention_id].append({
            'customer_id': customer_id,
            'timestamp': timestamp,
            'outcome': outcome
        })

    def get_intervention_success_rate(self, intervention_id: str) -> float:
        """
        Calculate the success rate of a specific intervention.
        
        :param intervention_id: Unique identifier for the intervention
        :return: Success rate as a float between 0 and 1
        """
        if intervention_id not in self.intervention_results:
            return 0.0
        
        results = self.intervention_results[intervention_id]
        success_count = sum(1 for result in results if result['outcome'] == 'success')
        return success_count / len(results)

    def get_intervention_summary(self, intervention_id: str) -> Dict:
        """
        Get a summary of the intervention results.
        
        :param intervention_id: Unique identifier for the intervention
        :return: Dictionary containing summary statistics
        """
        if intervention_id not in self.intervention_results:
            return {}
        
        results = self.intervention_results[intervention_id]
        df = pd.DataFrame(results)
        
        return {
            'total_applications': len(results),
            'success_rate': self.get_intervention_success_rate(intervention_id),
            'unique_customers': df['customer_id'].nunique(),
            'first_application': df['timestamp'].min(),
            'last_application': df['timestamp'].max()
        }

    def compare_interventions(self, intervention_ids: List[str]) -> pd.DataFrame:
        """
        Compare multiple interventions based on their success rates.
        
        :param intervention_ids: List of intervention IDs to compare
        :return: DataFrame with comparison results
        """
        comparison_data = []
        for intervention_id in intervention_ids:
            summary = self.get_intervention_summary(intervention_id)
            if summary:
                comparison_data.append({
                    'intervention_id': intervention_id,
                    'intervention_name': self.intervention_catalog.get_intervention(intervention_id)['name'],
                    'success_rate': summary['success_rate'],
                    'total_applications': summary['total_applications']
                })
        
        return pd.DataFrame(comparison_data).sort_values('success_rate', ascending=False)

    def get_customer_intervention_history(self, customer_id: str) -> pd.DataFrame:
        """
        Get the intervention history for a specific customer.
        
        :param customer_id: Unique identifier for the customer
        :return: DataFrame with the customer's intervention history
        """
        customer_history = []
        for intervention_id, results in self.intervention_results.items():
            for result in results:
                if result['customer_id'] == customer_id:
                    customer_history.append({
                        'intervention_id': intervention_id,
                        'intervention_name': self.intervention_catalog.get_intervention(intervention_id)['name'],
                        'timestamp': result['timestamp'],
                        'outcome': result['outcome']
                    })
        
        return pd.DataFrame(customer_history).sort_values('timestamp')