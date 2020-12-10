# Copyright 2020 Francesco Ceccon
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

"""Relaxation that replaces a convex function with an auxiliary variable."""

import math

import pyomo.environ as pe
from coramin.relaxations.custom_block import declare_custom_block
from coramin.relaxations.relaxations_base import BaseRelaxationData, ComponentWeakRef
from coramin.utils.coramin_enums import RelaxationSide
from pyomo.core.expr.visitor import identify_variables


@declare_custom_block(name='FactorableConvexExpressionRelaxation')
class FactorableConvexExpressionRelaxationData(BaseRelaxationData):
    def __init__(self, component):
        super().__init__(component)
        self._original_xs = None
        self._aux_var_ref = ComponentWeakRef(None)
        self._relaxed_f_x_expr = None
        self._original_f_x_expr = None

    @property
    def _aux_var(self):
        return self._aux_var_ref.get_component()

    def get_rhs_vars(self):
        return list(self._original_xs)

    def get_rhs_expr(self):
        return self._original_f_x_expr

    def vars_with_bounds_in_relaxation(self):
        return list()

    def set_input(self, aux_var, original_f_x_expr, relaxed_f_x_expr, persistent_solvers=None, large_eval_tol=math.inf,
                  use_linear_relaxation=True):
        relaxation_side = RelaxationSide.UNDER
        self._set_input(relaxation_side=relaxation_side, persistent_solvers=persistent_solvers,
                        use_linear_relaxation=use_linear_relaxation, large_eval_tol=large_eval_tol)

        self._original_f_x_expr = original_f_x_expr
        self._relaxed_f_x_expr = relaxed_f_x_expr

        self._original_xs = list(identify_variables(original_f_x_expr, include_fixed=False))
        self._aux_var_ref.set_component(aux_var)

    def build(self, aux_var, original_f_x_expr, relaxed_f_x_expr, persistent_solvers=None, large_eval_tol=math.inf,
              use_linear_relaxation=True):
        self.set_input(aux_var=aux_var, original_f_x_expr=original_f_x_expr, relaxed_f_x_expr=relaxed_f_x_expr,
                       persistent_solvers=persistent_solvers, large_eval_tol=large_eval_tol,
                       use_linear_relaxation=use_linear_relaxation)
        self.rebuild()

    def _build_relaxation(self):
        self._factored = pe.Constraint(expr=self._aux_var >= self._relaxed_f_x_expr)
        self._factored.construct()

    def is_rhs_convex(self):
        return True

    def is_rhs_concave(self):
        return False

    @property
    def use_linear_relaxation(self):
        return self._use_linear_relaxation

    @use_linear_relaxation.setter
    def use_linear_relaxation(self, value):
        self._use_linear_relaxation = value

    @property
    def relaxation_side(self):
        return RelaxationSide.UNDER
