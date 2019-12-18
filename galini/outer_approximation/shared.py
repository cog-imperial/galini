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
import numpy as np

from galini.util import solution_numerical_value
from galini.core import LinearExpression, Domain
from galini.cuts import CutType, Cut, CutsGenerator


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


def generate_cut(counter, i, constraint, x, w, x_k, fg, g_x):
    w[i] = 1.0
    d_fg = fg.reverse(1, w)
    w[i] = 0.0

    cut_name = _cut_name(counter, i, constraint.name)
    cut_expr = LinearExpression(x, d_fg, -np.dot(d_fg, x_k) + g_x[i])

    return Cut(
        CutType.LOCAL,
        cut_name,
        cut_expr,
        constraint.lower_bound,
        constraint.upper_bound,
        is_objective=False,
    )


def _cut_name(counter, i, name):
    return '_outer_approximation_{}_{}_r_{}'.format(
        i, name, counter
    )
