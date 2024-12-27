from sklearn.cluster import KMeans
import pandas as pd
import numpy as np

class CustomerSegmentation:
    def __init__(self, n_segments=5, predefined_segments=None):
        self.n_segments = n_segments
        self.predefined_segments = predefined_segments
        self.model = None
        self.segment_labels = None

    def fit(self, customer_data):
        """
        Fit the segmentation model to customer data.
        
        :param customer_data: DataFrame with customer features
        """
        if self.predefined_segments is not None:
            self._fit_predefined(customer_data)
        else:
            self._fit_kmeans(customer_data)

    def _fit_predefined(self, customer_data):
        """
        Assign customers to predefined segments.
        
        :param customer_data: DataFrame with customer features
        """
        if 'segment' not in customer_data.columns:
            raise ValueError("Customer data must include a 'segment' column for predefined segmentation.")
        
        self.segment_labels = customer_data['segment'].unique()
        self.n_segments = len(self.segment_labels)

    def _fit_kmeans(self, customer_data):
        """
        Fit KMeans model for automatic segmentation.
        
        :param customer_data: DataFrame with customer features
        """
        self.model = KMeans(n_clusters=self.n_segments)
        self.model.fit(customer_data)
        self.segment_labels = [f"Segment_{i}" for i in range(self.n_segments)]

    def predict(self, customer_data):
        """
        Predict segments for customer data.
        
        :param customer_data: DataFrame with customer features
        :return: array of segment labels
        """
        if self.predefined_segments is not None:
            if 'segment' not in customer_data.columns:
                raise ValueError("Customer data must include a 'segment' column for predefined segmentation.")
            return customer_data['segment'].values
        else:
            return np.array([self.segment_labels[i] for i in self.model.predict(customer_data)])

    def get_segment_profiles(self, customer_data):
        """
        Get profiles for each segment.
        
        :param customer_data: DataFrame with customer features
        :return: DataFrame with segment profiles
        """
        segments = self.predict(customer_data)
        customer_data_with_segments = customer_data.copy()
        customer_data_with_segments['segment'] = segments
        
        segment_profiles = customer_data_with_segments.groupby('segment').mean()
        return segment_profiles

    def set_predefined_segments(self, segment_labels):
        """
        Set predefined segments.
        
        :param segment_labels: List of segment labels
        """
        self.predefined_segments = segment_labels
        self.n_segments = len(segment_labels)
        self.segment_labels = segment_labels
        self.model = None  # Clear any existing KMeans model