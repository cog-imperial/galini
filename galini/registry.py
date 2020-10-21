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
"""Registry module."""
import abc
import sys
import pkg_resources


class Registry(metaclass=abc.ABCMeta):
    """Registry for pkg_resources entry points."""
    def __init__(self):
        self.group = self.group_name()
        self._registered = {}
        for entry_point in self.iter_entry_points():
            if entry_point.name in self._registered:
                print(
                    'Duplicate registered item {} found in {} registry.'.format(
                    entry_point.name, self.group)
                )
                sys.exit(1)
            obj_cls = entry_point.load()
            self._registered[entry_point.name] = obj_cls

    def get(self, name, default=None):
        """Return the entry point associated with name."""
        return self._registered.get(name, default)

    def __getitem__(self, item):
        """Return the entry point associated with name."""
        return self._registered[item]

    def keys(self):
        """Return the registered objects names."""
        return self._registered.keys()

    def items(self):
        """Return iterator over registered objects."""
        return self._registered.items()

    @abc.abstractmethod
    def group_name(self): # pragma: no cover
        """Return this registry group name."""
        pass

    def iter_entry_points(self): # pragma: no cover
        """Return an iterator over this registry entry points."""
        return pkg_resources.iter_entry_points(self.group_name())
