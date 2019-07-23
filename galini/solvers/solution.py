# Copyright 2017 Francesco Ceccon
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
"""Base class for solutions."""
import abc
import warnings
from collections import namedtuple


OptimalObjective = namedtuple('OptimalObjective', ['name', 'value'])
OptimalVariable = namedtuple('OptimalVariable', ['name', 'value'])


class Status(metaclass=abc.ABCMeta):
    """Solver status."""

    @abc.abstractmethod
    def is_success(self):
        """Predicate that return True if solve was successfull."""
        pass

    @abc.abstractmethod
    def is_infeasible(self):
        """Predicate that return True if problem is infeasible."""
        pass

    @abc.abstractmethod
    def is_unbounded(self):
        """Predicate that return True if problem is unbounded."""
        pass

    @abc.abstractmethod
    def description(self):
        """Return status description."""
        pass


class Solution(object):
    """Base class for all solutions.

    Solvers can subclass this class to add solver-specific information
    to the solution.
    """
    def __init__(self, status, optimal_obj=None, optimal_vars=None):
        if not isinstance(status, Status):
            raise TypeError('status must be subclass of Status')

        if optimal_vars is None:
            optimal_vars = []

        if isinstance(optimal_obj, list):
            if len(optimal_obj) != 1:
                raise ValueError('optimal_obj must be a list of length 1')
            warnings.warn('Multiple objectives not supported. Pass single value', DeprecationWarning)
            optimal_obj = optimal_obj[0]

        if optimal_obj is not None and not isinstance(optimal_obj, OptimalObjective):
            raise TypeError('optimal_obj must be OptimalObjective')

        for ov in optimal_vars:
            if not isinstance(ov, OptimalVariable):
                raise TypeError('optimal_vars must be collection of OptimalVariable')

        self.status = status
        self.objective = optimal_obj
        self.variables = optimal_vars

    @property
    def objectives(self):
        warnings.warn('Solution.objectives is deprecated. Use Solution.objective', DeprecationWarning)
        return [self.objective]

    def __str__(self):
        return 'Solution(status={}, objective={})'.format(
            self.status.description(), self.objective
        )

    def objective_value(self):
        if self.objective is None:
            return None
        return self.objective.value



class SolutionPool:
    """Contains a (bounded) solution pool, sorted by objective value.

    Parameters
    ----------
    n : int
        solution pool size
    """
    def __init__(self, n=5):
        self._pool = []
        self._n = n

    def add(self, solution):
        self._pool.append(_SolutionPoolSolution(solution))
        self._pool.sort()
        if len(self._pool) >= self._n:
            self._pool = self._pool[:self._n]

    def __len__(self):
        return len(self._pool)

    def __getitem__(self, idx):
        solution = self._pool[idx]
        return solution.inner

    def __iter__(self):
        return iter(self._pool)

    @property
    def head(self):
        if self._pool:
            return self._pool[0].inner
        return None


class _SolutionPoolSolution:
    def __init__(self, solution):
        self.inner = solution

    def __lt__(self, other):
        return self.inner.objective_value() < other.inner.objective_value()

    def __str__(self):
        return '_SolutionPoolSolution(objective_value={})'.format(self.inner.objective_value())

    def __repr__(self):
        return '<{} at {}>'.format(str(self), hex(id(self)))
