import pytest
from src.utils.config_loader import load_config


@pytest.fixture
def config():
    return load_config()
