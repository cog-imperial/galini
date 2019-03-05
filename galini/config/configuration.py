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



class _ConfigGroup(object):
    def __init__(self, name, items=None, strict=True):
        self.name = name
        if items is None:
            items = {}
        self._items = items
        self._strict = strict

    @classmethod
    def from_dict(cls, dict_):
        """Build _ConfigGroup from dictionary."""
        items = {}
        for key, value in dict_.items():
            if isinstance(value, dict):
                value = _ConfigGroup.from_dict(value)
            items[key] = value
        return cls(items)

    def keys(self):
        """Return group keys."""
        return self._items.keys()

    def items(self):
        """Return group items."""
        return self._items.items()

    def get(self, key, default=None):
        """Get value for key."""
        return self._items.get(key, default)

    def set(self, key, value):
        """Set value for key."""
        self._items.update({key: value})

    def pop(self, key, default=None):
        """Pop value for key."""
        return self._items.pop(key, default)

    def __getitem__(self, key):
        """Get value for key."""
        return self._items[key]

    def __getattr__(self, attr):
        return self._items[attr]

    def __contains__(self, name):
        return name in self._items

    def add_group(self, name, strict=True):
        """Add a new configuration group."""
        if name is self._items:
            raise ValueError('Group {} already present.'.format(name))
        group = _ConfigGroup(name, strict=strict)
        self._items[name] = group
        return group

    def update(self, other, path=None):
        """Update self with values from other."""
        for key, value in other.items():
            if path is None:
                sub_path = key
            else:
                sub_path = path + '.' + key

            if key not in self._items and self._strict:
                raise ValueError('Invalid configuration key/group "{}"'.format(sub_path))

            own_value = self._items.get(key, None)
            if isinstance(own_value, _ConfigGroup):
                own_value.update(value, path=sub_path)
            else:
                self._items[key] = value

    def __str__(self):
        key_values = ['{}: {}'.format(k, str(v)) for k, v in self._items.items()]
        return '{{{}}}'.format(', '.join(key_values))

    def __repr__(self):
        return '<{} at {}>'.format(str(self), hex(id(self)))


class GaliniConfig(object):
    """GALINI Configuration object."""
    def __init__(self):
        self._config = _ConfigGroup('root')

    def add_group(self, name, **kwargs):
        """Add a new configuration group."""
        return self._config.add_group(name, **kwargs)

    def get(self, key, default=None):
        """Get configuration value or group for key. Returns None if not present."""
        return self._config.get(key, default=default)

    def __getitem__(self, key):
        """Get configuration value or group for key. Raise KeyError if not present."""
        return self._config[key]

    def __getattr__(self, attr):
        """Get configuration value or group for key. Raise KeyError if not present."""
        return getattr(self._config, attr)

    def update(self, other):
        """Update config with other."""
        self._config.update(other)

    def __contains__(self, key):
        return key in self._config

    def __str__(self):
        return 'GaliniConfig({})'.format(str(self._config))

    def __repr__(self):
        return '<{} at {}>'.format(str(self), hex(id(self)))
