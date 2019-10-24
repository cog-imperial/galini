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
"""GALINI solve subcommand."""

import sys

from galini.commands import (
    CliCommandWithProblem,
    OutputTable,
    print_output_table,
    add_output_format_parser_arguments,
)
from galini.galini import Galini
from galini.logging import apply_config as apply_logging_config
from galini.timelimit import start_timelimit, set_timelimit

DEFAULT_SOLVER = 'bac'


class SolveCommand(CliCommandWithProblem):
    """Command to solve an optimization problem."""

    def execute_with_problem(self, model, problem, args):
        galini = Galini()
        if args.config:
            galini.update_configuration(args.config)

        solver_cls = galini.get_solver(args.solver.lower())

        if solver_cls is None:
            available = ', '.join(solvers_reg.keys())
            print(
                'Solver {} not available. Available solvers: {}'.format(
                args.solver, available)
            )
            sys.exit(1)

        apply_logging_config(galini.get_configuration_group('logging'))
        solver = solver_cls(galini)

        galini_group = galini.get_configuration_group('galini')
        timelimit = galini_group.get('timelimit')

        set_timelimit(timelimit)
        start_timelimit()

        solver.before_solve(model, problem)
        solution = solver.solve(
            problem,
            known_optimal_objective=args.known_optimal_objective
        )

        if solution is None:
            raise RuntimeError('Solver did not return a solution')

        obj_table = OutputTable('Objectives', [
            {'id': 'name', 'name': 'Objective', 'type': 't'},
            {'id': 'value', 'name': 'Value', 'type': 'f'},
        ])

        value = solution.objective.value
        if not problem.objective.original_sense.is_minimization():
            if value is not None:
                value = -value

        obj_table.add_row({
            'name': solution.objective.name,
            'value': value,
        })

        var_table = OutputTable('Variables', [
            {'id': 'name', 'name': 'Variable', 'type': 't'},
            {'id': 'value', 'name': 'Value', 'type': 'f'},
        ])
        for var in solution.variables:
            var_table.add_row({
                'name': var.name,
                'value': var.value,
            })

        counter_table = OutputTable('Counters', [
            {'id': 'name', 'name': 'Name', 'type': 't'},
            {'id': 'value', 'name': 'Value', 'type': 'f'},
        ])
        for counter in galini.telemetry.counters_values():
            counter_table.add_row(counter)

        print_output_table([obj_table, var_table, counter_table], args)


    def help_message(self):
        return "Solve a MINLP"

    def add_extra_parser_arguments(self, parser):
        parser.add_argument(
            '--solver',
            help='Specify the solver to use',
            default=DEFAULT_SOLVER,
        )
        parser.add_argument(
            '--config',
            help='Specify the configuration file',
        )
        parser.add_argument(
            '--known-optimal-objective',
            help='Specify the known optimal objective',
            type=float,
        )
        add_output_format_parser_arguments(parser)
