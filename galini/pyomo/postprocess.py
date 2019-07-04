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

"""Simple post-processing of problems."""
import galini.core as core
from galini.math import is_close, mc
from galini.util import expr_to_str


def detect_auxiliary_variables(problem):
    for constraint in problem.constraints:
        root_expr = constraint.root_expr

        if not isinstance(root_expr, core.SumExpression) or len(root_expr.children) != 2:
            continue

        a, b = root_expr.children
        if isinstance(a, core.QuadraticExpression) and isinstance(b, core.LinearExpression):
            quadratic = a
            linear = b
        elif isinstance(b, core.QuadraticExpression) and isinstance(a, core.LinearExpression):
            quadratic = b
            linear = a
        else:
            continue

        if len(linear.children) != 1 or len(quadratic.terms) != 1:
            continue

        if not is_close(linear.constant_term, 0.0, atol=mc.epsilon):
            continue

        var = linear.children[0]
        coef = linear.coefficient(var)
        term = quadratic.terms[0]

        if is_close(coef, -1.0, atol=mc.epsilon):
            if is_close(term.coefficient, 1.0, atol=mc.epsilon):
                var.reference = core.BilinearTermReference(term.var1, term.var2)

        if is_close(coef, 1.0, atol=mc.epsilon):
            if is_close(term.coefficient, -1.0, atol=mc.epsilon):
                var.reference = core.BilinearTermReference(term.var1, term.var2)
