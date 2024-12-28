import numpy as np
import pandas as pd
from typing import List, Dict, Any
from sklearn.inspection import PartialDependenceDisplay
import matplotlib.pyplot as plt
from shap import TreeExplainer, summary_plot

class ModelInterpreter:
    def __init__(self, model: Any, feature_names: List[str]):
        self.model = model
        self.feature_names = feature_names
        self.shap_explainer = None

    def feature_importance(self) -> pd.DataFrame:
        """
        Calculate feature importance if the model supports it.
        
        :return: DataFrame with feature importances
        """
        if hasattr(self.model, 'feature_importances_'):
            importances = self.model.feature_importances_
            return pd.DataFrame({
                'feature': self.feature_names,
                'importance': importances
            }).sort_values('importance', ascending=False)
        else:
            raise AttributeError("The model doesn't have feature_importances_ attribute.")

    def partial_dependence_plot(self, X: np.ndarray, features: List[int], feature_names: List[str]):
        """
        Create partial dependence plots for specified features.
        
        :param X: Input data
        :param features: List of feature indices to plot
        :param feature_names: Names of the features to plot
        """
        fig, ax = plt.subplots(figsize=(10, 6))
        display = PartialDependenceDisplay.from_estimator(
            self.model, X, features, feature_names=feature_names,
            ax=ax
        )
        plt.tight_layout()
        return fig

    def shap_summary_plot(self, X: np.ndarray):
        """
        Create a SHAP summary plot.
        
        :param X: Input data
        """
        if self.shap_explainer is None:
            self.shap_explainer = TreeExplainer(self.model)
        
        shap_values = self.shap_explainer.shap_values(X)
        fig = plt.figure()
        summary_plot(shap_values, X, feature_names=self.feature_names, show=False)
        plt.tight_layout()
        return fig

    def explain_prediction(self, instance: np.ndarray) -> Dict[str, float]:
        """
        Explain a single prediction using SHAP values.
        
        :param instance: Single instance to explain
        :return: Dictionary of feature names and their SHAP values
        """
        if self.shap_explainer is None:
            self.shap_explainer = TreeExplainer(self.model)
        
        shap_values = self.shap_explainer.shap_values(instance)
        return dict(zip(self.feature_names, shap_values[0]))