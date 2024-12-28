import yaml
from typing import Dict, Any

class Config:
    def __init__(self, config_path: str):
        self.config_path = config_path
        self.config_data = self.load_config()

    def load_config(self) -> Dict[str, Any]:
        """
        Load configuration from a YAML file.

        :return: Dictionary containing configuration data
        """
        with open(self.config_path, 'r') as config_file:
            return yaml.safe_load(config_file)

    def get(self, key: str, default: Any = None) -> Any:
        """
        Get a configuration value.

        :param key: Configuration key
        :param default: Default value if key is not found
        :return: Configuration value
        """
        return self.config_data.get(key, default)

    def set(self, key: str, value: Any):
        """
        Set a configuration value.

        :param key: Configuration key
        :param value: Configuration value
        """
        self.config_data[key] = value

    def save(self):
        """
        Save the current configuration to the YAML file.
        """
        with open(self.config_path, 'w') as config_file:
            yaml.dump(self.config_data, config_file)

    def __getitem__(self, key: str) -> Any:
        """
        Allow dictionary-like access to configuration values.

        :param key: Configuration key
        :return: Configuration value
        """
        return self.get(key)

    def __setitem__(self, key: str, value: Any):
        """
        Allow dictionary-like setting of configuration values.

        :param key: Configuration key
        :param value: Configuration value
        """
        self.set(key, value)