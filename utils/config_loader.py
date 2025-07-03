# utils/config_loader.py

import yaml
from pathlib import Path

class Config:
    def __init__(self, config_path="config/config.yaml"):
        self.config_path = Path(config_path)
        self._config = self._load_config()

    def _load_config(self):
        with open(self.config_path, "r") as f:
            return yaml.safe_load(f)

    def get(self, key_path, default=None):
        """
        İç içe geçmiş anahtarlara nokta ile erişmek için:
        config.get("ppo.learning_rate") → 0.0003
        """
        keys = key_path.split(".")
        val = self._config
        try:
            for k in keys:
                val = val[k]
            return val
        except (KeyError, TypeError):
            return default

    def all(self):
        return self._config
