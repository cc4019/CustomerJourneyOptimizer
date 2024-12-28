import pandas as pd
from typing import Dict, Any
from .config import Config

class DataLoader:
    def __init__(self, config: Config):
        self.config = config

    def load_customer_data(self) -> pd.DataFrame:
        """
        Load customer data from the configured source.

        :return: DataFrame containing customer data
        """
        data_path = self.config.get('customer_data_path')
        if data_path.endswith('.csv'):
            return pd.read_csv(data_path)
        elif data_path.endswith('.parquet'):
            return pd.read_parquet(data_path)
        else:
            raise ValueError(f"Unsupported file format for customer data: {data_path}")

    def load_intervention_data(self) -> pd.DataFrame:
        """
        Load intervention data from the configured source.

        :return: DataFrame containing intervention data
        """
        data_path = self.config.get('intervention_data_path')
        if data_path.endswith('.csv'):
            return pd.read_csv(data_path)
        elif data_path.endswith('.parquet'):
            return pd.read_parquet(data_path)
        else:
            raise ValueError(f"Unsupported file format for intervention data: {data_path}")

    def load_hva_data(self) -> pd.DataFrame:
        """
        Load High Value Action (HVA) data from the configured source.

        :return: DataFrame containing HVA data
        """
        data_path = self.config.get('hva_data_path')
        if data_path.endswith('.csv'):
            return pd.read_csv(data_path)
        elif data_path.endswith('.parquet'):
            return pd.read_parquet(data_path)
        else:
            raise ValueError(f"Unsupported file format for HVA data: {data_path}")

    def load_journey_data(self) -> pd.DataFrame:
        """
        Load customer journey data from the configured source.

        :return: DataFrame containing journey data
        """
        data_path = self.config.get('journey_data_path')
        if data_path.endswith('.csv'):
            return pd.read_csv(data_path)
        elif data_path.endswith('.parquet'):
            return pd.read_parquet(data_path)
        else:
            raise ValueError(f"Unsupported file format for journey data: {data_path}")

    def save_results(self, results: Dict[str, Any], filename: str):
        """
        Save analysis results to a file.

        :param results: Dictionary containing results to save
        :param filename: Name of the file to save results to
        """
        output_path = self.config.get('output_path')
        full_path = f"{output_path}/{filename}"
        
        if filename.endswith('.csv'):
            pd.DataFrame(results).to_csv(full_path, index=False)
        elif filename.endswith('.parquet'):
            pd.DataFrame(results).to_parquet(full_path, index=False)
        elif filename.endswith('.json'):
            pd.DataFrame(results).to_json(full_path, orient='records')
        else:
            raise ValueError(f"Unsupported file format for saving results: {filename}")
