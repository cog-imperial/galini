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
from argparse import ArgumentParser, Namespace
from galini.config import ConfigurationManager
from galini.commands import (
    CliCommandWithProblem,
    OutputTable,
    print_output_table,
    add_output_format_parser_arguments,
)
from galini.logging import RootLogger
from galini.solvers import SolversRegistry
from galini.timelimit import timeout


DEFAULT_SOLVER = 'oa'


class SolveCommand(CliCommandWithProblem):
    """Command to solve an optimization problem."""

    def execute_with_problem(self, problem, args):
        solvers_reg = SolversRegistry()
        solver_cls = solvers_reg.get(args.solver.lower())
        if solver_cls is None:
            available = ', '.join(solvers_reg.keys())
            print(
                'Solver {} not available. Available solvers: {}'.format(
                args.solver, available)
            )
            sys.exit(1)

        config_manager = ConfigurationManager()

        config_manager.initialize(solvers_reg, args.config)
        config = config_manager.configuration

        root_logger = RootLogger(config.logging)
        solver = solver_cls(config, solvers_reg)

        galini_group = config.get('galini')
        timelimit = galini_group.get('timelimit')
        with timeout(timelimit):
            solution = solver.solve(problem, logger=root_logger)

        if solution is None:
            raise RuntimeError('Solver did not return a solution')

        obj_table = OutputTable('Objectives', [
            {'id': 'name', 'name': 'Objective', 'type': 't'},
            {'id': 'value', 'name': 'Value', 'type': 'f'},
        ])
        for obj in solution.objectives:
            obj_table.add_row({
                'name': obj.name,
                'value': obj.value,
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

        print_output_table([obj_table, var_table], args)


    def help_message(self):
        return "Solve a MINLP"

    def add_extra_parser_arguments(self, parser):
        parser.add_argument('--solver', help='Specify the solver to use', default=DEFAULT_SOLVER)
        parser.add_argument('--config', help='Specify the configuration file')

        add_output_format_parser_arguments(parser)
