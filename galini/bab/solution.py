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

"""Branch & Bound solution."""
import heapq
from galini.solvers import Solution, Status


class BabStatus(Status):
    pass


class BabStatusSuccess(BabStatus):
    def is_success(self):
        return True

    def is_infeasible(self):
        return False

    def is_unbounded(self):
        return False

    def description(self):
        return 'Success'


class BabStatusInterrupted(BabStatus):
    def is_success(self):
        return False

    def is_infeasible(self):
        return False

    def is_unbounded(self):
        return False

    def description(self):
        return 'Interrupted'


class BabSolution(Solution):
    """Solution of the Branch & Bound algorithm."""
    def __init__(self, status, optimal_obj, optimal_vars, dual_bound,
                 nodes_visited=None, nodes_remaining=None, runtime=None,
                 is_timeout=None, has_converged=None, node_limit_exceeded=None):
        super().__init__(status, optimal_obj, optimal_vars)
        self.dual_bound = dual_bound
        self.nodes_visited = nodes_visited
        self.nodes_remaining = nodes_remaining
        self.runtime = runtime
        self.is_timeout = is_timeout
        self.has_converged = has_converged
        self.node_limit_exceeded = node_limit_exceeded



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
