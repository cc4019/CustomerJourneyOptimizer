import pandas as pd
import numpy as np
from causalimpact import CausalImpact

class InterventionToHVAAnalyzer:
    def __init__(self):
        pass

    def analyze_impact(self, data, intervention_start_date, intervention_end_date):
        """
        Analyze the causal impact of an intervention on HVAs.
        
        :param data: DataFrame with columns ['date', 'hva_count', 'other_features']
        :param intervention_start_date: Start date of the intervention
        :param intervention_end_date: End date of the intervention
        :return: CausalImpact object
        """
        pre_period = [data['date'].min(), intervention_start_date - pd.Timedelta(days=1)]
        post_period = [intervention_start_date, intervention_end_date]

        ci = CausalImpact(data, pre_period, post_period)
        return ci

    def summarize_impact(self, causal_impact):
        """
        Summarize the causal impact analysis.
        
        :param causal_impact: CausalImpact object
        :return: Dictionary with summary statistics
        """
        summary = causal_impact.summary()
        return {
            'average_effect': summary['average']['absolute']['actual'] - summary['average']['absolute']['predicted'],
            'cumulative_effect': summary['cumulative']['absolute']['actual'] - summary['cumulative']['absolute']['predicted'],
            'significance': summary['p'],
        }
