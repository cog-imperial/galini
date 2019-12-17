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

from galini.util import solution_numerical_value


def problem_is_linear(relaxed_problem):
    if relaxed_problem.objective.root_expr.polynomial_degree() > 1:
        return False

    for con in relaxed_problem.constraints:
        if con.root_expr.polynomial_degree() > 1:
            return False

    return True


def mip_variable_value(problem, sol):
    v = problem.variable_view(sol.name)
    return solution_numerical_value(
        sol,
        problem.lower_bound(v),
        problem.upper_bound(v),
    )
