import yaml
from .exceptions import ConfigError

class Config:
    def __init__(self, path="config.yaml"):
        try:
            with open(path, "r") as f:
                self.data = yaml.safe_load(f)
        except Exception as e:
            raise ConfigError(f"Failed to load config: {e}")

    @property
    def rules(self):
        return self.data.get("rules", {})

    @property
    def examples(self):
        return self.data.get("examples", {})