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

"""GALINI special_structure command."""
from suspect.convexity import Convexity
from suspect.monotonicity import Monotonicity
import pyomo.environ as pe
from galini.commands import (
    CliCommandWithProblem,
    OutputTable,
    add_output_format_parser_arguments,
    print_output_table,
)
from suspect.pyomo.quadratic import enable_standard_repn_for_quadratic_expression
from suspect.fbbt import perform_fbbt
from suspect.propagation import propagate_special_structure


enable_standard_repn_for_quadratic_expression()


def _cvx_to_str(cvx):
    return {
        Convexity.Linear: 'Linear',
        Convexity.Convex: 'Convex',
        Convexity.Concave: 'Concave',
    }.get(cvx, 'Unknown')


def _mono_to_str(mono):
    return {
        Monotonicity.Constant: 'Constant',
        Monotonicity.Nondecreasing: 'Nondecr.',
        Monotonicity.Nonincreasing: 'Nonincr.',
    }.get(mono, 'Unknown')


class SpecialStructureCommand(CliCommandWithProblem):
    def execute_with_model(self, model, args):
        bounds = perform_fbbt(model, max_iter=100)
        mono, cvx = propagate_special_structure(model, bounds)
        tables = []
        tables.append(self._output_variables(model, bounds, mono, cvx))
        tables.append(self._output_objectives(model, bounds, mono, cvx))
        tables.append(self._output_constraints(model, bounds, mono, cvx))
        print_output_table(tables, args)

    def _output_variables(self, model, bounds, _mono, _cvx):
        table = OutputTable('Variables', [
            {'id': 'name', 'name': 'Var.', 'type': 't'},
            {'id': 'domain', 'name': 'Dom.', 'type': 't'},
            {'id': 'lower_bound', 'name': 'LB', 'type': 'f'},
            {'id': 'upper_bound', 'name': 'UB', 'type': 'f'},
            {'id': 'original_lower_bound', 'name': 'OLB', 'type': 'f'},
            {'id': 'original_upper_bound', 'name': 'OUB', 'type': 'f'},
        ])
        for var in model.component_data_objects(pe.Var, active=True, descend_into=True):
            var_bounds = bounds[var]
            table.add_row({
                'name': var.name,
                'domain': var.domain,
                'lower_bound': var_bounds.lower_bound,
                'upper_bound': var_bounds.upper_bound,
                'original_lower_bound': var.lb,
                'original_upper_bound': var.ub,
            })
        return table

    def _output_objectives(self, model, bounds, mono, cvx):
        return self._output_expressions(
            model,
            bounds,
            mono,
            cvx,
            pe.Objective,
            lambda obj: obj.expr,
            'Obj.',
            'Objectives',
        )

    def _output_constraints(self, model, bounds, mono, cvx):
        return self._output_expressions(
            model,
            bounds,
            mono,
            cvx,
            pe.Constraint,
            lambda con: con.body,
            'Cons.',
            'Constraints',
        )

    def _output_expressions(self, model, bounds, mono, cvx, component_type, get_expr, type_, table_name):
        table = OutputTable(table_name, [
            {'id': 'name', 'name': type_, 'type': 't'},
            {'id': 'lower_bound', 'name': 'LB', 'type': 'f'},
            {'id': 'upper_bound', 'name': 'UB', 'type': 'f'},
            {'id': 'convexity', 'name': 'Cvx', 'type': 't'},
            {'id': 'monotonicity', 'name': 'Mono', 'type': 't'},
        ])
        for component in model.component_data_objects(component_type, active=True, descend_into=True):
            root_expr = get_expr(component)
            cvx_str = _cvx_to_str(cvx[root_expr])
            mono_str = _mono_to_str(mono[root_expr])
            expr_bounds = bounds[root_expr]
            table.add_row({
                'name': component.name,
                'lower_bound': expr_bounds.lower_bound,
                'upper_bound': expr_bounds.upper_bound,
                'convexity': cvx_str,
                'monotonicity': mono_str,
            })
        return table

    def help_message(self):
        return 'Print special structure information'

    def add_extra_parser_arguments(self, parser):
        add_output_format_parser_arguments(parser)
