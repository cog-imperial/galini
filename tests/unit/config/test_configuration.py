# pylint: skip-file
import pytest
from pathlib import Path
from galini.config import GaliniConfig


@pytest.fixture
def user_config():
    user_config_path = Path(__file__).parent / 'user_config.toml'
    return GaliniConfig(user_config_path)


def test_default_config():
    config = GaliniConfig()
    assert config
    assert config.logging['level'] == 'INFO'


def test_user_config_overrides_existing_keys(user_config):
    assert user_config.logging.level == 100


def test_user_config_can_have_extra_keys(user_config):
    assert user_config.ipopt.derivative_test == 'second-order'
