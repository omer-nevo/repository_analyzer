import yaml
from pathlib import Path

CONFIG_PATH = Path(__file__).resolve().parents[2] / "config" / "config.yaml"


def load_config():
    """Load configuration from YAML file."""
    with open(CONFIG_PATH, "r") as file:
        return yaml.safe_load(file)


def get_openai_key():
    """Returns openAi key from config"""
    config = load_config()
    return config.get("openai_api_key", None)


def get_rate_limiter_config():
    """Return rate limiter values from config (max_rate, time_period)."""
    config = load_config()
    max_rate = config.get("rate_limiter", {}).get("max_rate", 10)
    time_period = config.get("rate_limiter", {}).get("time_period", 1)
    return max_rate, time_period


def get_vector_db_config():
    """Return vector database properties (embedding_dim, chunk_size)."""
    config = load_config()
    embedding_dim = config.get("vector_db", {}).get("embedding_dim", 1536)
    chunk_size = config.get("vector_db", {}).get("chunk_size", 500)
    return embedding_dim, chunk_size
