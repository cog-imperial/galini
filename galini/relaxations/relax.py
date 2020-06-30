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

import pyomo.environ as pe
from coramin.relaxations.auto_relax import (
    RelaxationCounter,
    _relax_root_to_leaf_map,
    _relax_leaf_to_root_map,
    _relax_root_to_leaf_SumExpression,
    _relax_expr,
)
from coramin.utils.coramin_enums import RelaxationSide
from pyomo.core.expr.numvalue import polynomial_degree
from suspect.pyomo.quadratic import QuadraticExpression
from galini.relaxations.expressions import _relax_leaf_to_root_QuadraticExpression


_relax_leaf_to_root_map[QuadraticExpression] = _relax_leaf_to_root_QuadraticExpression
_relax_root_to_leaf_map[QuadraticExpression] = _relax_root_to_leaf_SumExpression


def relax(model, use_linear_relaxation=True):
    new_model = model.clone()

    original_to_new_var_map = pe.ComponentMap()

    for var in model.component_data_objects(pe.Var,
                                            active=True,
                                            descend_into=True):
        new_var = new_model.find_component(var.getname(fully_qualified=True))
        original_to_new_var_map[var] = new_var

    model = new_model

    aux_var_map = dict()

    degree_map = pe.ComponentMap()

    model.relaxations = pe.Block()
    model.aux_vars = pe.VarList()
    model.aux_cons = pe.ConstraintList()

    counter = RelaxationCounter()

    for obj in model.component_data_objects(ctype=pe.Objective, active=True):
        degree = polynomial_degree(obj.expr)
        if degree is not None:
            if degree <= 1:
                continue

        assert obj.is_minimizing()

        relaxation_side = RelaxationSide.UNDER
        relaxation_side = RelaxationSide.BOTH
        relaxation_side_map = pe.ComponentMap()
        relaxation_side_map[obj.expr] = relaxation_side

        new_body = _relax_expr(expr=obj.expr, aux_var_map=aux_var_map, parent_block=model,
                               relaxation_side_map=relaxation_side_map, counter=counter, degree_map=degree_map)
        obj._expr = new_body

    for cons in model.component_data_objects(ctype=pe.Constraint, active=True):
        body_degree = polynomial_degree(cons.body)
        if body_degree is not None:
            if body_degree <= 1:
                continue

        if cons.has_lb() and cons.has_ub():
            relaxation_side = RelaxationSide.BOTH
        elif cons.has_lb():
            relaxation_side = RelaxationSide.OVER
        elif cons.has_ub():
            relaxation_side = RelaxationSide.UNDER
        else:
            raise ValueError('Encountered a constraint without a lower or an upper bound: ' + str(c))

        relaxation_side = RelaxationSide.BOTH
        relaxation_side_map = pe.ComponentMap()
        relaxation_side_map[cons.body] = relaxation_side

        new_body = _relax_expr(expr=cons.body, aux_var_map=aux_var_map, parent_block=model,
                               relaxation_side_map=relaxation_side_map, counter=counter, degree_map=degree_map)
        cons._body = new_body

    reverse_var_map = dict()
    for var in model.component_data_objects(pe.Var, active=True):
        reverse_var_map[id(var)] = var

    var_relax_map = pe.ComponentMap()

    for aux_var_info, aux_var_value in aux_var_map.items():
        _, relaxation = aux_var_value
        if len(aux_var_info) == 2 and aux_var_info[1] == 'quadratic':
            var = reverse_var_map[aux_var_info[0]]
            vars = [var]
        elif len(aux_var_info) == 3 and aux_var_info[2] == 'mul':
            var0 = reverse_var_map[aux_var_info[0]]
            var1 = reverse_var_map[aux_var_info[1]]
            vars = [var0, var1]
        else:
            raise RuntimeError("Invalid aux var info ", aux_var_info, aux_var_value)

        for var in vars:
            if var not in var_relax_map:
                var_relax_map[var] = [relaxation]
            else:
                var_relax_map[var].append(relaxation)

    for _, relaxation in aux_var_map.values():
        relaxation.use_linear_relaxation = use_linear_relaxation
        relaxation.rebuild()
    return model, var_relax_map, original_to_new_var_map