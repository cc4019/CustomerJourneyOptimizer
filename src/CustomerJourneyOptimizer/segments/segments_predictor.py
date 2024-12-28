
import numpy as np
import pandas as pd
from typing import List, Dict, Any
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

class SegmentsPredictor:
    def __init__(self, n_estimators: int = 100, random_state: int = 42):
        self.model = RandomForestClassifier(n_estimators=n_estimators, random_state=random_state)
        self.feature_encoder = LabelEncoder()
        self.segment_encoder = LabelEncoder()
        self.features = None

    def fit(self, X: pd.DataFrame, y: pd.Series):
        """
        Fit the segments predictor to customer data.

        :param X: DataFrame of customer features
        :param y: Series of customer segments
        """
        self.features = X.columns.tolist()

        # Encode categorical features
        X_encoded = X.copy()
        for column in X.columns:
            if X[column].dtype == 'object':
                X_encoded[column] = self.feature_encoder.fit_transform(X[column])

        # Encode segments
        y_encoded = self.segment_encoder.fit_transform(y)

        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(X_encoded, y_encoded, test_size=0.2, random_state=42)

        # Fit the model
        self.model.fit(X_train, y_train)

        # Evaluate the model
        y_pred = self.model.predict(X_test)
        print(classification_report(y_test, y_pred, target_names=self.segment_encoder.classes_))

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Predict segments for given customer data.

        :param X: DataFrame of customer features
        :return: Array of predicted segments
        """
        if self.model is None:
            raise ValueError("Model has not been fitted. Call fit() first.")

        # Ensure all expected features are present
        missing_features = set(self.features) - set(X.columns)
        if missing_features:
            raise ValueError(f"Missing features: {missing_features}")

        # Encode categorical features
        X_encoded = X.copy()
        for column in self.features:
            if X[column].dtype == 'object':
                X_encoded[column] = self.feature_encoder.transform(X[column])

        # Make predictions
        predictions_encoded = self.model.predict(X_encoded)
        return self.segment_encoder.inverse_transform(predictions_encoded)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """
        Predict segment probabilities for given customer data.

        :param X: DataFrame of customer features
        :return: Array of segment probabilities
        """
        if self.model is None:
            raise ValueError("Model has not been fitted. Call fit() first.")

        # Ensure all expected features are present
        missing_features = set(self.features) - set(X.columns)
        if missing_features:
            raise ValueError(f"Missing features: {missing_features}")

        # Encode categorical features
        X_encoded = X.copy()
        for column in self.features:
            if X[column].dtype == 'object':
                X_encoded[column] = self.feature_encoder.transform(X[column])

        # Make predictions
        return self.model.predict_proba(X_encoded)

    def get_feature_importance(self) -> pd.DataFrame:
        """
        Get the importance of each feature in predicting segments.

        :return: DataFrame of feature importances
        """
        if self.model is None:
            raise ValueError("Model has not been fitted. Call fit() first.")

        importances = self.model.feature_importances_
        feature_importance = pd.DataFrame({
            'feature': self.features,
            'importance': importances
        }).sort_values('importance', ascending=False)

        return feature_importance

    def explain_prediction(self, customer_data: pd.DataFrame) -> Dict[str, Any]:
        """
        Explain the segment prediction for a single customer.

        :param customer_data: DataFrame containing a single customer's data
        :return: Dictionary containing the predicted segment and feature contributions
        """
        if self.model is None:
            raise ValueError("Model has not been fitted. Call fit() first.")

        if len(customer_data) != 1:
            raise ValueError("customer_data should contain exactly one row")

        # Predict segment
        predicted_segment = self.predict(customer_data)[0]

        # Get feature importances
        feature_importance = self.get_feature_importance()

        # Calculate feature contributions
        contributions = {}
        for feature in self.features:
            if customer_data[feature].dtype == 'object':
                value = self.feature_encoder.transform(customer_data[feature])[0]
            else:
                value = customer_data[feature].values[0]
            importance = feature_importance.loc[feature_importance['feature'] == feature, 'importance'].values[0]
            contributions[feature] = value * importance

        return {
            'predicted_segment': predicted_segment,
            'feature_contributions': contributions
        }