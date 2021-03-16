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

from galini.commands import (
    CliCommandWithProblem,
    OutputTable,
    print_output_table,
    add_output_format_parser_arguments,
)
from galini.galini import Galini


DEFAULT_ALGORITHM = 'bac'


class SolveCommand(CliCommandWithProblem):
    """Command to solve an optimization problem."""

    def execute_with_model(self, model, args):
        galini = Galini()

        if args.config:
            galini.update_configuration(args.config)

        solution = galini.solve(model, args.algorithm, known_optimal_objective=args.known_optimal_objective)

        status_table = OutputTable('Solution', [
            {'id': 'status', 'name': 'Status', 'type': 't'}
        ])

        if solution is None:
            status_table.add_row({'status': 'unboundedOrInfeasible'})
            print_output_table([status_table], args)
            return

        status_table.add_row({'status': solution.status.description()})

        obj_table = OutputTable('Objectives', [
            {'id': 'name', 'name': 'Objective', 'type': 't'},
            {'id': 'value', 'name': 'Value', 'type': 'f'},
        ])

        obj_table.add_row({
            'name': 'objective',
            'value': solution.objective,
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

        print_output_table([status_table, obj_table, var_table, counter_table], args)

    def help_message(self):
        return "Solve a MINLP"

    def add_extra_parser_arguments(self, parser):
        parser.add_argument(
            '--algorithm',
            help='Specify the algorithm to use',
            default=DEFAULT_ALGORITHM,
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
