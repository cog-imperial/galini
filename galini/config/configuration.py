# Copyright 2018 Francesco Ceccon
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""GALINI Configuration module."""
from typing import Dict, Any
import pkg_resources
import toml


class ConfigGroupNotFound(Exception):
    """Exception thrown if no configuration group was found."""
    def __init__(self, group_name: str) -> None:
        super().__init__()
        self.group_name = group_name


class GaliniConfig(object):
    """GALINI Configuration object."""

    def __init__(self, user_config_path: str = None) -> None:
        default_config_path = 'default.toml'
        template = pkg_resources.resource_string(__name__, default_config_path)
        default_config = toml.loads(template.decode('utf-8'))
        if user_config_path:
            user_config = toml.load(user_config_path)
        else:
            user_config = {}
        self._config = {**default_config, **user_config}

    def get_group(self, name: str) -> Dict[str, Any]:
        """Return a configuration group."""
        try:
            group = self._config[name]
            if not isinstance(group, dict):
                raise RuntimeError('not a group')
            return group
        except KeyError:
            raise ConfigGroupNotFound(name)
