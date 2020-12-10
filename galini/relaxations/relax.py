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
import numpy as np
from coramin.relaxations.auto_relax import (
    RelaxationCounter,
    _relax_root_to_leaf_map,
    _relax_leaf_to_root_map,
    _relax_root_to_leaf_SumExpression,
    _relax_expr,
)
from coramin.relaxations import PWXSquaredRelaxation, PWUnivariateRelaxation
from coramin.utils.coramin_enums import RelaxationSide
from pyomo.core.expr.numvalue import polynomial_degree
from suspect.pyomo.quadratic import QuadraticExpression
from galini.relaxations.expressions import _relax_leaf_to_root_QuadraticExpression


_relax_leaf_to_root_map[QuadraticExpression] = _relax_leaf_to_root_QuadraticExpression
_relax_root_to_leaf_map[QuadraticExpression] = _relax_root_to_leaf_SumExpression


class RelaxationData:
    def __init__(self, model):
        self.model = model
        self.original_to_new_var_map = pe.ComponentMap()
        self.aux_var_map = dict()
        self.reverse_var_map = dict()
        self.var_relax_map = pe.ComponentMap()
        self.degree_map = pe.ComponentMap()
        self.counter = RelaxationCounter()


def relax_expression(model, expr, relaxation_side, data):
    relaxation_side_map = pe.ComponentMap()
    relaxation_side_map[expr] = relaxation_side

    expr = _relax_expr(expr=expr, aux_var_map=data.aux_var_map, parent_block=model,
                       relaxation_side_map=relaxation_side_map, counter=data.counter, degree_map=data.degree_map)
    return expr


def relax_inequality(model, ineq_expr, relaxation_side, data):
    if ineq_expr.nargs() == 3:
        lb, expr, ub = ineq_expr.args
    else:
        assert ineq_expr.nargs() == 2
        c1, c2 = ineq_expr.args
        if type(c1) in pe.nonpyomo_leaf_types or not c1.is_expression_type():
            assert c2.is_expression_type()
            lb = c1
            expr = c2
            ub = None
        elif type(c2) in pe.nonpyomo_leaf_types or not c2.is_expression_type():
            assert c1.is_expression_type()
            lb = None
            expr = c1
            ub = c2
        else:
            raise ValueError('Cannot handle inequality expression {} with args {}, {}'.format(ineq_expr, type(c1), type(c2)))
    relaxed_expr = relax_expression(model, expr, relaxation_side, data)
    return pe.inequality(lb, relaxed_expr, ub)


def relax_constraint(model, cons, data, inplace=False):
    body_degree = polynomial_degree(cons.body)
    if body_degree is not None:
        if body_degree <= 1:
            return pe.Constraint(expr=cons.body)

    if cons.has_lb() and cons.has_ub():
        relaxation_side = RelaxationSide.BOTH
    elif cons.has_lb():
        relaxation_side = RelaxationSide.OVER
    elif cons.has_ub():
        relaxation_side = RelaxationSide.UNDER
    else:
        raise ValueError('Encountered a constraint without a lower or an upper bound: ' + str(cons))

    relaxation_side = RelaxationSide.BOTH
    new_body = relax_expression(model, cons.body, relaxation_side, data)

    if inplace:
        cons._body = new_body
        return cons

    lb, ub = cons.lb, cons.ub
    if cons.has_lb() and cons.has_ub():
        assert np.isclose(lb, ub)
        return pe.Constraint(expr=new_body == lb)
    elif cons.has_lb():
        return pe.Constraint(expr=lb <= new_body)
    elif cons.has_ub():
        return pe.Constraint(expr=new_body <= ub)

    raise ValueError('Encountered a constraint without a lower or an upper bound: ' + str(cons))


def update_relaxation_data(model, data):
    for var in model.component_data_objects(pe.Var, active=True, descend_into=True):
        data.reverse_var_map[id(var)] = var

    for aux_var_info, aux_var_value in data.aux_var_map.items():
        _, relaxation = aux_var_value
        aux_var_info_len = len(aux_var_info)
        if aux_var_info_len == 2 and aux_var_info[1] == 'quadratic':
            var = data.reverse_var_map[aux_var_info[0]]
            vars = [var]
        elif aux_var_info_len == 3 and aux_var_info[2] == 'mul':
            var0 = data.reverse_var_map[aux_var_info[0]]
            var1 = data.reverse_var_map[aux_var_info[1]]
            vars = [var0, var1]
        elif aux_var_info_len == 3 and aux_var_info[2] in ['pow', 'div']:
            vars = []
        elif aux_var_info_len == 2 and aux_var_info[1] == 'exp':
            vars = []
        else:
            raise RuntimeError("Invalid aux var info ", aux_var_info, aux_var_value)

        for var in vars:
            if var not in data.var_relax_map:
                data.var_relax_map[var] = [relaxation]
            else:
                data.var_relax_map[var].append(relaxation)


def rebuild_relaxations(model, data, use_linear_relaxation=True):
    for _, relaxation in data.aux_var_map.values():
        relaxation.use_linear_relaxation = use_linear_relaxation
        relaxation.rebuild()


def relax(model, data, use_linear_relaxation=True):
    new_model = model.clone()

    for var in model.component_data_objects(pe.Var,
                                            active=True,
                                            descend_into=True):
        new_var = new_model.find_component(var.getname(fully_qualified=True))
        data.original_to_new_var_map[var] = new_var

    model = new_model

    model.relaxations = pe.Block()
    model.aux_vars = pe.VarList()
    model.aux_cons = pe.ConstraintList()

    for obj in model.component_data_objects(ctype=pe.Objective, active=True):
        degree = polynomial_degree(obj.expr)
        if degree is not None:
            if degree <= 1:
                continue

        assert obj.is_minimizing()

        # relaxation_side = RelaxationSide.UNDER
        relaxation_side = RelaxationSide.BOTH

        new_body = relax_expression(model, obj.expr, relaxation_side, data)
        obj._expr = new_body

    for cons in model.component_data_objects(ctype=pe.Constraint, active=True):
        relax_constraint(model, cons, data, inplace=True)

    update_relaxation_data(model, data)
    rebuild_relaxations(model, data, use_linear_relaxation)

    return model