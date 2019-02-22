# pylint: skip-file
import pytest
from pathlib import Path
from galini.config import GaliniConfig, ConfigurationManager
from galini.solvers import SolversRegistry


@pytest.fixture
def user_config():
    user_config_path = Path(__file__).parent / 'user_config.toml'
    manager = ConfigurationManager()
    solvers_reg = SolversRegistry()
    manager.initialize(solvers_reg, str(user_config_path))
    return manager.configuration


def test_default_config(user_config):
    assert user_config.logging['stdout']


def test_user_config_overrides_existing_keys(user_config):
    assert user_config.logging.level == 100


def test_user_config_can_have_extra_keys(user_config):
    assert user_config.ipopt.ipopt.derivative_test == 'second-order'
