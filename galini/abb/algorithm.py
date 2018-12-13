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

"""Alpha BB algorithm."""
import numpy as np
import datetime
import pytimeparse
from galini.bab import BabTree, KSectionBranchingStrategy
from galini.abb.relaxation import AlphaBBRelaxation


class NodeSelectionStrategy(object):
    def __init__(self):
        self.nodes = []

    def insert_node(self, node):
        self.nodes.append(node)

    def next_node(self):
        return self.nodes.pop()


class BabAlgorithm(object):
    def __init__(self):
        self._init()

    def _init(self):
        self.problem_lower_bound = -np.inf
        self.problem_upper_bound = np.inf
        self.tolerance = 1e-5
        self.start_time = None
        self.timeout = datetime.timedelta(seconds=pytimeparse.parse('1 min'))

    def has_converged(self):
        return (self.problem_upper_bound - self.problem_lower_bound) < self.tolerance

    def has_timeout(self):
        if self.start_time is None:
            return False
        now = datetime.datetime.now()
        return (now - self.start_time) > self.timeout

    def start_now(self):
        self.start_time = datetime.datetime.now()

    def solve(self, problem):
        branching_strategy = KSectionBranchingStrategy(2)
        node_selection_strategy = NodeSelectionStrategy()
        tree = BabTree(branching_strategy, node_selection_strategy)

        self.start_now()

        root_solution = self.solve_root_problem(problem)
        tree.add_root(problem, root_solution)

        if self.has_converged():
            # problem is convex so it has converged already
            return root_solution

        while not self.has_converged() and not self.has_timeout():
            current_node = tree.next_node()
            print(current_node.coordinate, current_node.solution)
            if current_node is None:
                raise RuntimeError('not converged')
            node_children = current_node.branch()

            for child in node_children:
                print(child.variable.name)
                solution = self.solve_problem(child.problem)
                child.solution = solution
                tree.insert_node(child)
        print(current_node.solution)
        return current_node.solution

    def solve_root_problem(self, problem):
        return self.solve_problem(problem)

    def solve_problem(self, problem):
        raise NotImplementedError()


class AlphaBBAlgorithm(BabAlgorithm):
    def __init__(self, nlp_solver, minlp_solver, solver_name, run_id):
        super().__init__()
        self._nlp_solver = nlp_solver
        self._minlp_solver = minlp_solver

        self._solver_name = solver_name
        self._run_id = run_id

    def solve_problem(self, problem):
        relaxation = AlphaBBRelaxation()
        relaxed_problem = relaxation.relax(problem)
        solution = self._minlp_solver.solve(relaxed_problem)
        print(solution.objectives[0].value, [v for v in solution.variables])
        assert len(solution.objectives) == 1
        relaxed_obj_value = solution.objectives[0].value

        self.problem_lower_bound = max(self.problem_lower_bound, relaxed_obj_value)

        x = [v.value for v in solution.variables]

        fg = problem.expression_tree_data().eval(x, [obj.root_expr.idx for obj in problem.objectives])
        fg_x = fg.forward(0, x)[0]

        self.problem_upper_bound = min(self.problem_upper_bound, fg_x)

        print(self.problem_lower_bound, self.problem_upper_bound)

        return solution
