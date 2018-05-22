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
from typing import Any, Dict, Iterable
import abc
import sys
import pkg_resources
import galini.logging as log


class Registry(metaclass=abc.ABCMeta):
    """Registry for pkg_resources entry points."""
    def __init__(self) -> None:
        self.group = self.group_name()
        self._registered: Dict[str, Any] = {}
        for entry_point in pkg_resources.iter_entry_points(self.group_name()):
            if entry_point.name in self._registered:
                log.error(
                    'Duplicate registered item %s found in %s registry.',
                    entry_point.name, self.group,
                )
                sys.exit(1)
            obj_cls = entry_point.load()
            self._registered[entry_point.name] = obj_cls

    def get(self, name: str, default: Any = None) -> Any:
        """Return entry point associated with name."""
        return self._registered.get(name, default)

    def keys(self) -> Iterable[str]:
        """Return the registered objects names."""
        return self._registered.keys()

    @abc.abstractmethod
    def group_name(self) -> str:
        """Return this registry group name."""
        pass
