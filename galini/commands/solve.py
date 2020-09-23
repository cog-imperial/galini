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

import numpy as np
import pyomo.environ as pe

from galini.commands import (
    CliCommandWithProblem,
    OutputTable,
    print_output_table,
    add_output_format_parser_arguments,
)
from galini.galini import Galini
from galini.timelimit import seconds_elapsed_since, current_time


DEFAULT_ALGORITHM = 'bac'


class SolveCommand(CliCommandWithProblem):
    """Command to solve an optimization problem."""

    def execute_with_model(self, model, args):
        galini = Galini()

        if args.config:
            galini.update_configuration(args.config)

        algo_cls = galini.get_algorithm(args.algorithm.lower())

        if algo_cls is None:
            available = ', '.join(galini.available_algorithms())
            print(
                'Algorithm {} not available. Available algorithms: {}'.format(
                args.algorithm, available)
            )
            sys.exit(1)

        galini_group = galini.get_configuration_group('galini')
        timelimit = galini_group.get('timelimit')
        elapsed_counter = galini.telemetry.create_gauge('elapsed_time', 0.0)

        galini.timelimit.set_timelimit(timelimit)
        galini.timelimit.start_now()

        start_time = current_time()

        # Check problem only has one objective, if it's maximisation convert it to minimisation
        original_objective = None
        for objective in model.component_data_objects(pe.Objective, active=True):
            if original_objective is not None:
                raise ValueError('Algorithm does not support models with multiple objectives')
            original_objective = objective

        if original_objective is None:
            model._objective = pe.Objective(expr=0.0, sense=pe.minimize)
        else:
            if not original_objective.is_minimizing():
                new_objective = pe.Objective(expr=-original_objective.expr, sense=pe.minimize)
            else:
                new_objective = pe.Objective(expr=original_objective.expr, sense=pe.minimize)
            model._objective = new_objective
            model._objective.is_originally_minimizing = original_objective.is_minimizing()
            original_objective.deactivate()

        for var in model.component_data_objects(pe.Var, active=True):
            lb = var.lb if var.lb is not None else -np.inf
            ub = var.ub if var.ub is not None else np.inf
            value = var.value
            if value is not None and (value < lb or value > ub):
                if np.isinf(lb) or np.isinf(ub):
                    value = 0.0
                else:
                    value = lb + (ub - lb) / 2.0
                    if var.is_integer() or var.is_binary():
                        value = np.rint(value)

                var.set_value(value)

        algo = algo_cls(galini)
        solution = algo.solve(
            model,
            known_optimal_objective=args.known_optimal_objective
        )

        del model._objective
        if original_objective is not None:
            original_objective.activate()

        elapsed_counter.set_value(seconds_elapsed_since(start_time))

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

        value = solution.objective
        if original_objective is not None:
            if not original_objective.is_minimizing():
                if value is not None:
                    value = -value

            obj_table.add_row({
                'name': original_objective.name,
                'value': value,
            })
        else:
            obj_table.add_row({
                'name': None,
                'value': 0.0,
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
