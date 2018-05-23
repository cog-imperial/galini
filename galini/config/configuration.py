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
from typing import Any, Dict, Iterable, Tuple
import pkg_resources
import toml


class _ConfigGroup(object):
    def __init__(self, items: Dict[str, Any]) -> None:
        self._items = items

    @classmethod
    def from_dict(cls, dict_: Dict[str, Any]) -> '_ConfigGroup':
        """Build _ConfigGroup from dictionary."""
        items = {}
        for key, value in dict_.items():
            if isinstance(value, dict):
                value = _ConfigGroup.from_dict(value)
            items[key] = value
        return cls(items)

    def keys(self) -> Iterable[str]:
        """Return group keys."""
        return self._items.keys()

    def items(self) -> Iterable[Tuple[str, Any]]:
        """Return group items."""
        return self._items.items()

    def get(self, key: str) -> Any:
        """Get value for key."""
        return self._items.get(key)

    def __getitem__(self, key: str) -> Any:
        """Get value for key."""
        return self._items[key]

    def __getattr__(self, attr: str) -> Any:
        return self._items[attr]

    def update(self, other: '_ConfigGroup') -> None:
        """Update self with values from other."""
        for key, value in other.items():
            if isinstance(value, _ConfigGroup):
                grp = self._items[key]
                if not isinstance(grp, _ConfigGroup):
                    self._items[key] = value
                else:
                    grp.update(value)
            else:
                current_value = self._items.get(key)
                if current_value and isinstance(current_value, _ConfigGroup):
                    raise RuntimeError('Trying to set configuration group to value.')
                self._items[key] = value


class _Config(object):
    def __init__(self, path: str) -> None:
        self._root = _ConfigGroup.from_dict(toml.load(path))

    def keys(self) -> Iterable[str]:
        """Return config keys."""
        return self._root.keys()

    def items(self) -> Iterable[Tuple[str, Any]]:
        """Return config items."""
        return self._root.items()

    def get(self, key: str) -> Any:
        """Get config key."""
        return self._root.get(key)

    def __getitem__(self, key: str) -> Any:
        return self._root[key]

    def __getattr__(self, attr: str) -> Any:
        return getattr(self._root, attr)

    def update(self, other: '_Config') -> None:
        """Update config with other."""
        # pylint: disable=protected-access
        self._root.update(other._root)


class GaliniConfig(object):
    """GALINI Configuration object."""

    def __init__(self, user_config_path: str = None) -> None:
        default_config_path = 'default.toml'
        template = pkg_resources.resource_filename(__name__, default_config_path)
        default_config = _Config(template)
        if user_config_path:
            user_config = _Config(str(user_config_path))
            default_config.update(user_config)
        self._config = default_config

    def get(self, key: str) -> Any:
        """Get configuration value or group for key. Returns None if not present."""
        return self._config.get(key)

    def __getitem__(self, key: str) -> Any:
        """Get configuration value or group for key. Raise KeyError if not present."""
        return self._config[key]

    def __getattr__(self, attr: str) -> Any:
        """Get configuration value or group for key. Raise KeyError if not present."""
        return getattr(self._config, attr)
