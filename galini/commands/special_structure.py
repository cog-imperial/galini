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
from texttable import Texttable
from suspect.convexity import Convexity
from suspect.monotonicity import Monotonicity
from galini.commands import CliCommandWithProblem
from galini.special_structure import detect_special_structure


def _cvx_to_str(cvx):
    return {
        Convexity.Linear: 'Linear',
        Convexity.Convex: 'Convex',
        Convexity.Concave: 'Concave',
    }.get(cvx, '')


def _mono_to_str(mono):
    return {
        Monotonicity.Constant: 'Constant',
        Monotonicity.Nondecreasing: 'Nondecr.',
        Monotonicity.Nonincreasing: 'Nonincr.',
    }.get(mono, '')


class SpecialStructureCommand(CliCommandWithProblem):
    def execute_with_problem(self, problem, args):
        ctx = detect_special_structure(problem)

        self._print_variables(problem, ctx)
        print()
        self._print_objectives(problem, ctx)
        print()
        self._print_constraints(problem, ctx)

    def _print_variables(self, problem, ctx):
        bounds = ctx.bounds
        table = Texttable()
        table.set_cols_dtype(['t', 't', 'f', 'f'])
        table.set_deco(Texttable.HEADER)
        table.header(['Var.', 'Dom.', 'LB', 'UB'])
        domain_to_str = ['R', 'I', 'B']
        for var_name in problem.variables():
            var = problem.variable(var_name)
            var_bounds = bounds(var.variable)
            var_domain = domain_to_str[var.domain]
            table.add_row([var_name, var_domain, var_bounds.lower_bound, var_bounds.upper_bound])
        print(table.draw())

    def _print_objectives(self, problem, ctx):
        self._print_expressions(problem, ctx, problem.objectives.items(), 'Obj.')

    def _print_constraints(self, problem, ctx):
        self._print_expressions(problem, ctx, problem.constraints.items(), 'Cons.')

    def _print_expressions(self, problem, ctx, expressions, type_):
        table = Texttable()
        table.header([type_, 'LB', 'UB', 'Cvx', 'Mono'])
        table.set_deco(Texttable.HEADER)
        table.set_cols_dtype(['t', 'f', 'f', 't', 't'])
        for expr_name, expr in expressions:
            root_expr = expr.root_expr
            cvx = _cvx_to_str(ctx.convexity(root_expr))
            mono = _mono_to_str(ctx.monotonicity(root_expr))
            bounds = ctx.bounds(root_expr)
            table.add_row([expr_name, bounds.lower_bound, bounds.upper_bound, cvx, mono])
        print(table.draw())

    def help_message(self):
        return 'Print special structure information'

    def add_extra_parser_arguments(self, parser):
        pass
