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
import numpy as np
import galini.core as core
from galini.math import is_close, mc
from galini.util import expr_to_str


def detect_auxiliary_variables(problem):
    bilinear_aux_variables = dict()
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
                bilinear_aux_variables[(term.var1.idx, term.var2.idx)] = var

        if is_close(coef, 1.0, atol=mc.epsilon):
            if is_close(term.coefficient, -1.0, atol=mc.epsilon):
                var.reference = core.BilinearTermReference(term.var1, term.var2)
                bilinear_aux_variables[(term.var1.idx, term.var2.idx)] = var

    problem.metadata['bilinear_aux_variables'] = bilinear_aux_variables


def detect_rlt_constraints(problem):
    possible_rlt = dict()
    seen_sum = set()

    for constraint in problem.constraints:
        root_expr = constraint.root_expr
        cons_lb = constraint.lower_bound
        if cons_lb is None:
            cons_lb = -np.inf
        cons_ub = constraint.upper_bound
        if cons_ub is None:
            cons_ub = np.inf

        bounds_zero = (
            is_close(cons_lb, 0.0, atol=mc.epsilon) and
            is_close(cons_ub, 0.0, atol=mc.epsilon)
        )

        # If it's a linear expression it can be the summation to 1 or RLT
        # IF it's bilinear it can be RLT
        # if it's product it can be RLT
        if isinstance(root_expr, core.LinearExpression):
            if bounds_zero:
                is_rlt, var, summed_vars = _detect_rlt_expression_linear(root_expr)
            else:
                is_rlt = False

            bounds_one = (
                is_close(cons_lb, 1.0, atol=mc.epsilon) and
                is_close(cons_ub, 1.0, atol=mc.epsilon)
            )

            if bounds_one:
                # Check all coefficients
                is_sum = True
                for v in root_expr.children:
                    if not is_close(root_expr.coefficient(v), 1.0, atol=mc.epsilon):
                        is_sum = False
                if is_sum:
                    summed_vars_idx = tuple(sorted(v.idx for v in root_expr.children))
                    seen_sum.add(summed_vars_idx)

        elif isinstance(root_expr, core.SumExpression):
            if not bounds_zero:
                continue
            is_rlt, var, summed_vars = _detect_rlt_expression_product(root_expr)
            if not is_rlt:
                is_rlt, var, summed_vars = _detect_rlt_expression_bilinear(root_expr)

        else:
            is_rlt = False

        if not is_rlt:
            continue

        summed_vars_idx = tuple(sorted(v.idx for v in summed_vars))
        if summed_vars_idx not in possible_rlt:
            possible_rlt[summed_vars_idx] = []
        possible_rlt[summed_vars_idx].append((constraint, var, summed_vars))

        # constraint.metadata['is_rlt_constraint'] = (non_aux_variable, aux_variables)
    if len(possible_rlt) < len(seen_sum):
        for var_indexes, constraints in possible_rlt.items():
            if var_indexes in seen_sum:
                # it's a RLT
                for constraint, var, aux_vars in constraints:
                    constraint.metadata['rlt_constraint_info'] = (var, aux_vars)
    else:
        for var_indexes in seen_sum:
            constraints = possible_rlt.get(var_indexes, [])
            for constraint, var, aux_vars in constraints:
                constraint.metadata['rlt_constraint_info'] = (var, aux_vars)


def _detect_rlt_expression_product(root_expr):
    """Matches expression of type x1 - x1*(x2+x3+...+xn)"""
    return False, None, None


def _detect_rlt_expression_bilinear(root_expr):
    """Matches expression of type x1 - x1x2 - x1x3 - ... - x1xn"""
    if len(root_expr.children) != 2:
        return False, None, None

    a, b = root_expr.children
    matches, var, var_sign, quadratic = _variable_and_quadratic(a, b)
    if not matches:
        matches, var, var_sign, quadratic = _variable_and_quadratic(b, a)
        if not matches:
            return False, None, None

    sum_vars = []
    for term in quadratic.terms:
        if not is_close(term.coefficient, -1.0 * var_sign, atol=mc.epsilon):
            return False, None, None
        if term.var1 != var and term.var2 != var:
            return False, None, None
        if term.var1 == var:
            sum_vars.append(term.var2)
        else:
            sum_vars.append(term.var1)

    return True, var, sum_vars


def _variable_and_quadratic(a, b):
    if isinstance(a, core.LinearExpression):
        if len(a.children) != 1:
            return False, None, None, None

        if not isinstance(b, core.QuadraticExpression):
            return False, None, None, None

        var = a.children[0]
        coef = a.coefficient(var)

        if not is_close(np.abs(coef), 1.0, atol=mc.epsilon):
            return False, None, None, None

        coef_sign = np.sign(coef)
        return True, var, coef_sign, b

    elif isinstance(a, core.Variable):
        if not isinstance(b, core.QuadraticExpression):
            return False, None, None

        return True, a, 1.0, b

    return False, None, None, None


def _detect_rlt_expression_linear(root_expr):
    """Matches expression of type x1 - w12 - w13 - ... - w1n"""
    aux_variables = [v for v in root_expr.children if v.is_auxiliary]
    non_aux_variables = [v for v in root_expr.children if not v.is_auxiliary]

    if len(non_aux_variables) != 1:
        return False, None, None

    non_aux_variable = non_aux_variables[0]
    non_aux_coef = root_expr.coefficient(non_aux_variable)
    non_aux_coef_is_one = is_close(np.abs(non_aux_coef), 1.0, atol=mc.epsilon)

    if not non_aux_coef_is_one:
        return False, None, None

    non_aux_coef_sign = np.sign(non_aux_coef)

    sum_vars = []
    for var in aux_variables:
        ref = var.reference

        if not ref:
            return False, None, None

        # Check coef is 1.0...
        coef = root_expr.coefficient(var)
        if not is_close(np.abs(coef), 1.0, atol=mc.epsilon):
            return False, None, None

        # ... and opposite sign of aux var
        if np.sign(coef) != -1.0 * non_aux_coef_sign:
            return False, None, None

        # ... and one aux variable is the non aux one
        if ref.var1 != non_aux_variable and ref.var2 != non_aux_variable:
            return False, None, None

        if ref.var1 == non_aux_variable:
            sum_vars.append(ref.var2)

        if ref.var2 == non_aux_variable:
            sum_vars.append(ref.var1)

    return True, non_aux_variable, sum_vars
