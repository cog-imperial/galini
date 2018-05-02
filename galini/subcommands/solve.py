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

import logging
import sys
from galini.config import GaliniConfig
from galini.subcommands import CliCommand
from galini.solvers import SolversRegistry
from galini.mip import MIPSolverRegistry
from galini.nlp import NLPSolverRegistry
from galini.pyomo import read_pyomo_model, dag_from_pyomo_model


DEFAULT_SOLVER = 'oa'


class SolveCommand(CliCommand):
    """Command to solve an optimization problem."""

    def execute(self, args):
        solvers_reg = SolversRegistry()
        solver_cls = solvers_reg.get(args.solver.lower())
        if solver_cls is None:
            available = ', '.join(solvers_reg.keys())
            logging.error(
                'Solver %s not available. Available solvers: %s',
                args.solver, available
            )
            sys.exit(1)

        config = GaliniConfig(args.config)
        mip_solver_registry = MIPSolverRegistry()
        nlp_solver_registry = NLPSolverRegistry()
        solver = solver_cls(config, mip_solver_registry, nlp_solver_registry)

        pyomo_model = read_pyomo_model(args.problem)
        dag = dag_from_pyomo_model(pyomo_model)
        solver.solve(dag)

    def help_message(self):
        return "Solve a MINLP"

    def add_parser_arguments(self, parser):
        parser.add_argument('problem')
        parser.add_argument('--solver', help='Specify the solver to use', default=DEFAULT_SOLVER)
        parser.add_argument('--config', help='Specify the configuration file')
