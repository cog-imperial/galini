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

"""Branch & Bound node selection strategy."""
import abc
import heapq


class NodeSelectionStrategy(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def insert_node(self, node):
        """Insert `node` in the tree."""
        pass

    @abc.abstractmethod
    def has_nodes(self):
        """Return True if there are mode nodes to visit."""
        pass

    @abc.abstractmethod
    def next_node(self):
        """Return next node."""
        pass


class BestLowerBoundSelectionStrategy(NodeSelectionStrategy):
    class _Node(object):
        def __init__(self, node):
            self.inner = node

        def __lt__(self, other):
            if self.inner.has_solution:
                self_state = self.inner.state
            else:
                assert self.inner.has_parent
                self_state = self.inner.parent.state

            if other.inner.has_solution:
                other_state = other.inner.state
            else:
                assert other.inner.has_parent
                other_state = other.inner.parent.state

            self_lb = self_state.lower_bound_solution
            other_lb = other_state.lower_bound_solution

            if not self_lb.status.is_success():
                return False
            if not other_lb.status.is_success():
                return True
            return self_lb.objective_value() < other_lb.objective_value()

    def __init__(self, algorithm):
        self.nodes = []

    def insert_node(self, node):
        heapq.heappush(self.nodes, self._Node(node))

    def has_nodes(self):
        return len(self.nodes) > 0

    def next_node(self):
        if len(self.nodes) == 0:
            return None
        node = heapq.heappop(self.nodes)
        return node.inner
