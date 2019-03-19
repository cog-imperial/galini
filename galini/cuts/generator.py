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
"""Cuts generator interface."""
import abc
from enum import Enum


class CutType(Enum):
    GLOBAL = 1
    LOCAL = 2


class Cut(object):
    def __init__(self, type_, name, expr, lower_bound, upper_bound):
        if not isinstance(type_, CutType):
            raise ValueError('type_ must be a valid CutType')
        self.type_ = type_
        self.name = name
        self.expr = expr
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound

    @property
    def is_global(self):
        return self._type == CutType.GLOBAL

    @property
    def is_local(self):
        return self._type == CutType.LOCAL


class CutsGenerator(metaclass=abc.ABCMeta):
    def __init__(self, config):
        pass

    def before_start_at_root(self, run_id, problem, relaxed_problem):
        pass

    def after_end_at_root(self, run_id, problem, relaxed_problem, solution):
        pass

    def before_start_at_node(self, run_id, problem, relaxed_problem):
        pass

    def after_end_at_node(self, run_id, problem, relaxed_problem, solution):
        pass

    @abc.abstractmethod
    def generate(self, run_id, problem, relaxed_problem, mip_solution, tree, node):
        pass
