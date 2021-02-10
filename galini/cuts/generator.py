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


class Cut:
    """Represent a cut to be added to the problem."""

    def __init__(self, type_, name, expr, lower_bound, upper_bound,
                 is_objective=False):
        if not isinstance(type_, CutType):
            raise ValueError('type_ must be a valid CutType')
        self.type_ = type_
        self.name = name
        self.expr = expr
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound
        self.is_objective = is_objective
        self._index = None

    @property
    def index(self):
        """Get cut index in cut pool."""
        return self._index

    @index.setter
    def index(self, index):
        """Set index after adding to cut pool."""
        if self._index is not None:
            raise RuntimeError('Trying to set index of already indexed cut')
        self._index = index

    @property
    def is_global(self):
        """Return `True` if cut is global."""
        return self.type_ == CutType.GLOBAL

    @property
    def is_local(self):
        """Return `True` if cut is local."""
        return self.type_ == CutType.LOCAL


class CutsGenerator(metaclass=abc.ABCMeta):
    """CutsGenerator interface."""

    description = None

    def __init__(self, galini, config):
        pass

    def before_start_at_root(self, problem, relaxed_problem):
        """Called before the cut loop starts at root node."""
        pass

    def after_end_at_root(self, problem, relaxed_problem, solution):
        """Called after the cut loop at root node."""
        pass

    def before_start_at_node(self, problem, relaxed_problem):
        """Called before the cut loop starts at any non-root node."""
        pass

    def after_end_at_node(self, problem, relaxed_problem, solution):
        """Called after the cut loop at any non-root node."""
        pass

    def has_converged(self, state):
        """Predicate that returns `True` if the cut generator finished."""
        pass

    @abc.abstractmethod
    def generate(self, problem, relaxed_problem, mip_solution, tree, node):
        """Generate new cuts."""
        pass
