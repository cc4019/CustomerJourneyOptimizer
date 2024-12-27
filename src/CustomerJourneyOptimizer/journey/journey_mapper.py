import pandas as pd
import numpy as np
from sklearn.cluster import KMeans

class JourneyMapper:
    def __init__(self, n_clusters=5):
        self.n_clusters = n_clusters
        self.cluster_model = KMeans(n_clusters=self.n_clusters)

    def fit(self, journey_data):
        """
        Fit the journey mapper to the data.
        
        :param journey_data: DataFrame with columns ['customer_id', 'timestamp', 'action']
        """
        # Aggregate actions by customer and time
        journey_matrix = self._create_journey_matrix(journey_data)
        
        # Fit the clustering model
        self.cluster_model.fit(journey_matrix)

    def transform(self, journey_data):
        """
        Transform journey data into cluster assignments.
        
        :param journey_data: DataFrame with columns ['customer_id', 'timestamp', 'action']
        :return: DataFrame with columns ['customer_id', 'timestamp', 'cluster']
        """
        journey_matrix = self._create_journey_matrix(journey_data)
        clusters = self.cluster_model.predict(journey_matrix)
        
        return pd.DataFrame({
            'customer_id': journey_data['customer_id'].unique(),
            'cluster': clusters
        })

    def _create_journey_matrix(self, journey_data):
        """
        Create a matrix representation of customer journeys.
        
        :param journey_data: DataFrame with columns ['customer_id', 'timestamp', 'action']
        :return: numpy array with shape (n_customers, n_actions)
        """
        pivot = journey_data.pivot_table(
            index='customer_id', 
            columns='action', 
            values='timestamp', 
            aggfunc='count',
            fill_value=0
        )
        return pivot.values