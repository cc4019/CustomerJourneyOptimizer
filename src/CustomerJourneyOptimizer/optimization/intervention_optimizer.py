import numpy as np
from scipy.optimize import linear_sum_assignment

class InterventionOptimizer:
    def __init__(self, interventions, customer_segments):
        self.interventions = interventions
        self.customer_segments = customer_segments
        self.impact_matrix = np.zeros((len(interventions), len(customer_segments)))

    def set_impact(self, intervention_index, segment_index, impact):
        """
        Set the impact of an intervention on a customer segment.
        
        :param intervention_index: Index of the intervention
        :param segment_index: Index of the customer segment
        :param impact: Impact value
        """
        self.impact_matrix[intervention_index, segment_index] = impact

    def optimize(self):
        """
        Optimize intervention assignments to customer segments.
        
        :return: List of (intervention, segment) pairs
        """
        row_ind, col_ind = linear_sum_assignment(self.impact_matrix, maximize=True)
        
        assignments = []
        for i, j in zip(row_ind, col_ind):
            assignments.append((self.interventions[i], self.customer_segments[j]))
        
        return assignments
