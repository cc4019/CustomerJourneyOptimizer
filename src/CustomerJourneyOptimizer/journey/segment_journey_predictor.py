import numpy as np
import pandas as pd
from typing import List, Dict, Any
from sklearn.preprocessing import LabelEncoder

class SegmentJourneyPredictor:
    def __init__(self):
        self.transition_matrix = None
        self.segment_encoder = LabelEncoder()
        self.segments = None

    def fit(self, journey_data: pd.DataFrame):
        """
        Fit the journey predictor to historical journey data.

        :param journey_data: DataFrame with columns ['customer_id', 'timestamp', 'segment']
        """
        # Encode segments
        self.segment_encoder.fit(journey_data['segment'])
        journey_data['encoded_segment'] = self.segment_encoder.transform(journey_data['segment'])

        # Get unique segments
        self.segments = self.segment_encoder.classes_

        # Initialize transition matrix
        n_segments = len(self.segments)
        self.transition_matrix = np.zeros((n_segments, n_segments))

        # Compute transition probabilities
        for customer_id in journey_data['customer_id'].unique():
            customer_journey = journey_data[journey_data['customer_id'] == customer_id].sort_values('timestamp')
            for i in range(len(customer_journey) - 1):
                from_segment = customer_journey.iloc[i]['encoded_segment']
                to_segment = customer_journey.iloc[i+1]['encoded_segment']
                self.transition_matrix[from_segment, to_segment] += 1

        # Normalize transition probabilities
        row_sums = self.transition_matrix.sum(axis=1)
        self.transition_matrix = self.transition_matrix / row_sums[:, np.newaxis]

    def predict_next_segment(self, current_segment: str) -> str:
        """
        Predict the next most likely segment given the current segment.

        :param current_segment: Current segment of the customer
        :return: Predicted next segment
        """
        if self.transition_matrix is None:
            raise ValueError("Model has not been fitted. Call fit() first.")

        current_encoded = self.segment_encoder.transform([current_segment])[0]
        next_encoded = np.argmax(self.transition_matrix[current_encoded])
        return self.segment_encoder.inverse_transform([next_encoded])[0]

    def predict_journey(self, start_segment: str, n_steps: int) -> List[str]:
        """
        Predict a customer's journey for a given number of steps.

        :param start_segment: Starting segment of the customer
        :param n_steps: Number of steps to predict
        :return: List of predicted segments
        """
        if self.transition_matrix is None:
            raise ValueError("Model has not been fitted. Call fit() first.")

        journey = [start_segment]
        current_segment = start_segment

        for _ in range(n_steps):
            next_segment = self.predict_next_segment(current_segment)
            journey.append(next_segment)
            current_segment = next_segment

        return journey

    def segment_transition_probabilities(self, segment: str) -> Dict[str, float]:
        """
        Get the transition probabilities from a given segment to all other segments.

        :param segment: The segment to get transition probabilities for
        :return: Dictionary of segment transition probabilities
        """
        if self.transition_matrix is None:
            raise ValueError("Model has not been fitted. Call fit() first.")

        segment_index = self.segment_encoder.transform([segment])[0]
        probabilities = self.transition_matrix[segment_index]
        return dict(zip(self.segments, probabilities))

    def most_likely_paths(self, start_segment: str, n_steps: int, top_k: int = 3) -> List[Dict[str, Any]]:
        """
        Find the top-k most likely paths from a given starting segment.

        :param start_segment: Starting segment
        :param n_steps: Number of steps in the path
        :param top_k: Number of top paths to return
        :return: List of dictionaries containing path and probability
        """
        if self.transition_matrix is None:
            raise ValueError("Model has not been fitted. Call fit() first.")

        start_index = self.segment_encoder.transform([start_segment])[0]
        
        # Initialize paths with start segment
        paths = [([start_index], 1.0)]
        
        for _ in range(n_steps):
            new_paths = []
            for path, prob in paths:
                current = path[-1]
                for next_segment, trans_prob in enumerate(self.transition_matrix[current]):
                    new_paths.append((path + [next_segment], prob * trans_prob))
            
            # Keep only top-k paths
            paths = sorted(new_paths, key=lambda x: x[1], reverse=True)[:top_k]
        
        # Convert indices back to segment names
        result = []
        for path, prob in paths:
            segment_path = self.segment_encoder.inverse_transform(path)
            result.append({
                'path': list(segment_path),
                'probability': prob
            })
        
        return result
