"""GALINI Configuration module."""
import pkg_resources
import toml


class ConfigGroupNotFound(Exception):
    def __init__(self, group_name):
        self.group_name = group_name


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

    def get_group(self, name):
        try:
            group = self._config[name]
            if not isinstance(group, dict):
                raise RuntimeError('not a group')
            return group
        except KeyError:
            raise ConfigGroupNotFound(name)
