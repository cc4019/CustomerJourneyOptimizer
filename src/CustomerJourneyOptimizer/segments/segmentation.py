from sklearn.cluster import KMeans
import pandas as pd

class CustomerSegmentation:
    def __init__(self, n_segments=5):
        self.n_segments = n_segments
        self.model = KMeans(n_clusters=self.n_segments)

    def fit(self, customer_data):
        """
        Fit the segmentation model to customer data.
        
        :param customer_data: DataFrame with customer features
        """
        self.model.fit(customer_data)

    def predict(self, customer_data):
        """
        Predict segments for customer data.
        
        :param customer_data: DataFrame with customer features
        :return: array of segment labels
        """
        return self.model.predict(customer_data)

