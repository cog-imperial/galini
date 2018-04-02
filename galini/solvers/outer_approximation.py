from galini.solvers import MINLPSolver


class OuterApproximationSolver(MINLPSolver):
    def __init__(self, config, mip_solver_registry, nlp_solver_registry):
        print(config)
