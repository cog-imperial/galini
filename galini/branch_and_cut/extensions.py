#  Copyright 2020 Francesco Ceccon
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.

"""Extensions registries used in the branch and cut algorithm"""

from galini.registry import Registry


class InitialPrimalSearchStrategyRegistry(Registry):
    """Registry of initial primal search strategies."""
    def group_name(self):
        return 'galini.initial_primal_search'


class PrimalHeuristicRegistry(Registry):
    """Registry of primal heuristics."""
    def group_name(self):
        return 'galini.primal_heuristic'


class NodeSelectionStrategyRegistry(Registry):
    """Registry of node selection strategies."""
    def group_name(self):
        return 'galini.node_selection_strategy'


class BranchingStrategyRegistry(Registry):
    """Registry of branching strategies."""
    def group_name(self):
        return 'galini.branching_strategy'


class RelaxationRegistry(Registry):
    """Registry of relaxations."""
    def group_name(self):
        return 'galini.relaxation'
