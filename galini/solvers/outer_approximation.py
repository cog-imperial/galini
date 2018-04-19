from galini.solvers import MINLPSolver


class OuterApproximationSolver(MINLPSolver):
    def __init__(self, config, _mip_solver_registry, nlp_solver_registry):
        self._nlp_solver_cls = nlp_solver_registry.get('ipopt')
        if self._nlp_solver_cls is None:
            raise RuntimeError('ipopt solver is required for OuterApproximationSolver')
        self._config = config

    def solve(self, problem):
        nlp_solver = self._nlp_solver_cls(self._config)
        nlp_solver.solve(problem)
