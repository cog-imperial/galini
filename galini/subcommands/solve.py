"""GALINI solve subcommand."""
import logging
import sys
from galini.config import GaliniConfig
from galini.subcommands import CliCommand
from galini.solvers import SolversRegistry
from galini.mip import MIPSolversRegistry
from galini.nlp import NLPSolversRegistry
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
        mip_solver_registry = MIPSolversRegistry()
        nlp_solver_registry = NLPSolversRegistry()
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
