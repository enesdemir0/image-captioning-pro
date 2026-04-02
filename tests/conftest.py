import pytest
from src.utils.config_loader import load_config

@pytest.fixture(scope="session")
def config():
    """Provides the config dictionary to all tests."""
    return load_config()