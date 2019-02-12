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
from galini.commands import (
    CliCommandWithProblem,
    OutputTable,
    add_output_format_parser_arguments,
    print_output_table,
)
from galini.special_structure import detect_special_structure


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
    def execute_with_problem(self, problem, args):
        ctx = detect_special_structure(problem)

        tables = []
        tables.append(self._output_variables(problem, ctx))
        tables.append(self._output_objectives(problem, ctx))
        tables.append(self._output_constraints(problem, ctx))
        print_output_table(tables, args)

    def _output_variables(self, problem, ctx):
        bounds = ctx.bounds
        table = OutputTable('Variables', [
            {'id': 'name', 'name': 'Var.', 'type': 't'},
            {'id': 'domain', 'name': 'Dom.', 'type': 't'},
            {'id': 'lower_bound', 'name': 'LB', 'type': 'f'},
            {'id': 'upper_bound', 'name': 'UB', 'type': 'f'},
            {'id': 'original_lower_bound', 'name': 'OLB', 'type': 'f'},
            {'id': 'original_upper_bound', 'name': 'OUB', 'type': 'f'},
        ])
        for variable in problem.variables:
            var = problem.variable_view(variable)
            var_bounds = bounds(var.variable)
            var_domain = var.domain.name[0]
            table.add_row({
                'name': variable.name,
                'domain': var.domain,
                'lower_bound': var_bounds.lower_bound,
                'upper_bound': var_bounds.upper_bound,
                'original_lower_bound': variable.lower_bound,
                'original_upper_bound': variable.upper_bound,
            })
        return table

    def _output_objectives(self, problem, ctx):
        return self._output_expressions(
            problem,
            ctx,
            problem.objectives,
            'Obj.',
            'Objectives',
        )

    def _output_constraints(self, problem, ctx):
        return self._output_expressions(
            problem,
            ctx,
            problem.constraints,
            'Cons.',
            'Constraints',
        )

    def _output_expressions(self, problem, ctx, expressions, type_, table_name):
        table = OutputTable(table_name, [
            {'id': 'name', 'name': type_, 'type': 't'},
            {'id': 'lower_bound', 'name': 'LB', 'type': 'f'},
            {'id': 'upper_bound', 'name': 'UB', 'type': 'f'},
            {'id': 'convexity', 'name': 'Cvx', 'type': 't'},
            {'id': 'monotonicity', 'name': 'Mono', 'type': 't'},
        ])
        for expr in expressions:
            root_expr = expr.root_expr
            cvx = _cvx_to_str(ctx.convexity(root_expr))
            mono = _mono_to_str(ctx.monotonicity(root_expr))
            bounds = ctx.bounds(root_expr)
            table.add_row({
                'name': expr.name,
                'lower_bound': bounds.lower_bound,
                'upper_bound': bounds.upper_bound,
                'convexity': cvx,
                'monotonicity': mono,
            })
        return table

    def help_message(self):
        return 'Print special structure information'

    def add_extra_parser_arguments(self, parser):
        add_output_format_parser_arguments(parser)
