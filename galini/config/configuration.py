"""GALINI Configuration module."""
import pkg_resources
import toml


class GaliniConfig(object):
    """GALINI Configuration object."""

    def __init__(self, user_config_path=None):
        default_config_path = 'default.toml'
        template = pkg_resources.resource_string(__name__, default_config_path)
        default_config = toml.loads(template.decode('utf-8'))
        if user_config_path:
            user_config = toml.load(user_config_path)
        else:
            user_config = {}
        self._config = {**default_config, **user_config}
