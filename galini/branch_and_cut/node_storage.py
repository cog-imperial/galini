#  Copyright 2019 Francesco Ceccon
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

"""Branch & Cut node storage. Contains original and convex problem."""

from galini.branch_and_bound.branching import branch_at_point


class NodeStorage:
    def __init__(self, problem, convex_problem):
        self.problem = problem
        self.convex_problem = convex_problem

    def branching_data(self):
        return self.problem

    def branch_at_point(self, branching_point):
        problem_children = branch_at_point(self.problem, branching_point)
        convex_problem_children = [None] * len(problem_children)
        assert len(problem_children) == len(convex_problem_children)

        return [
            NodeStorage(problem, convex_problem)
            for problem, convex_problem
            in zip(problem_children, convex_problem_children)
        ]
