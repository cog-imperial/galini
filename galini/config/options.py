# Copyright 2019 Francesco Ceccon
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
"""Solver options."""
import abc


class OptionsGroup(object):
    def __init__(self, name, options=None):
        if options is None:
            options = []
        self._options = options
        self.name = name

    def add(self, option):
        self._options.add(option)

    def iter(self):
        return iter(self._options)


class SolverOptions(OptionsGroup):
    pass


class CutsGeneratorOptions(OptionsGroup):
    pass


class ExternalSolverOptions(OptionsGroup):
    def __init__(self, name):
        super().__init__(name, None)


class Option(metaclass=abc.ABCMeta):
    def __init__(self, name, default=None, description=None):
        self.name = name
        self.default = default
        self.description = description
        self.value = None

    @abc.abstractmethod
    def is_valid(self):
        pass


class NumericOption(Option):
    def __init__(self, name, min_value=None, max_value=None,
                 default=None, description=None):
        super().__init__(name, default, description)
        self.min_value = min_value
        self.max_value = max_value

    def is_valid(self):
        if self.min_value is not None:
            if self.value < self.min_value:
                return False
        if self.max_value is not None:
            if self.value > self.max_value:
                return False
        return True


class IntegerOption(Option):
    def __init__(self, name, min_value=None, max_value=None,
                 default=None, description=None):
        super().__init__(name, default, description)
        self.min_value = min_value
        self.max_value = max_value

    def is_valid(self):
        if self.min_value is not None:
            if self.value < self.min_value:
                return False
        if self.max_value is not None:
            if self.value > self.max_value:
                return False
        return isinstance(self.value, (int,))


class BoolOption(Option):
    def is_valid(self):
        return isinstance(self.value, bool)


class StringOption(Option):
    def is_valid(self):
        return isinstance(self.value, str)


class StringListOption(Option):
    def is_valid(self):
        return isinstance(self.value, list)


class EnumOption(Option):
    def __init__(self, name, values=None,
                 default=None, description=None):
        super().__init__(name, default, description)
        self.values = values

    def is_valid(self):
        if self.values is not None:
            if self.value not in self.values:
                return False
        return isinstance(self.value, str)
