# Copyright 2019 Francesco Ceccon
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
"""Wrapper around pulp cplex interface."""
import pulp
from galini.logging import INFO, WARNING, ERROR
from galini.timelimit import seconds_left


class CplexSolver:
    """Wrapper around pulp.CPLEX_PY to integrate with GALINI."""
    def __init__(self, logger, run_id, galini):
        self._inner = pulp.CPLEX_PY()
        self._logger = logger
        self._run_id = run_id
        self._config = galini.get_configuration_group('mip.cplex')

    def available(self):
        """Predicate to check if the solver is available."""
        return self._inner.available()

    def actualSolve(self, lp): # pylint: disable=invalid-name,missing-docstring
        import cplex
        # Same as CPLEX_PY.actualSolve, but overrides log settings
        # and fixes some bugs
        self._inner.buildSolverModel(lp)

        self._setup_logs()

        model = lp.solverModel
        # set problem as mip if all continuous variables
        is_mip = True
        for var_type in model.variables.get_types():
            if var_type != 'C':
                is_mip = False
                break

        if is_mip:
            model.set_problem_type(cplex.Cplex.problem_type.LP)

        self._inner.callSolver(lp)
        solution_status = self._inner.findSolutionValues(lp)
        for var in lp.variables():
            var.modified = False
        for constraint in lp.constraints.values():
            constraint.modified = False

        # because of a bug in pulp, need to assign dual values here
        try:
            if model.get_problem_type() == cplex.Cplex.problem_type.LP:
                cons_name = [cons for cons in lp.constraints]
                constraintpivalues = dict(zip(cons_name,
                                              lp.solverModel.solution.get_dual_values(cons_name)))
                lp.assignConsPi(constraintpivalues)
        except cplex.exceptions.CplexSolverError:
            pass
        return solution_status

    def _setup_logs(self):
        if not self.available():
            return
        model = self._inner.solverModel
        model.set_warning_stream(_CplexLoggerAdapter(self._logger, self._run_id, WARNING))
        model.set_error_stream(_CplexLoggerAdapter(self._logger, self._run_id, ERROR))
        model.set_log_stream(_CplexLoggerAdapter(self._logger, self._run_id, INFO))
        model.set_results_stream(_CplexLoggerAdapter(self._logger, self._run_id, INFO))

        for key, value in self._config.items():
            if isinstance(value, dict):
                raise ValueError("Invalid CPLEX parameter '{}'".format(key))

            # Parameters are specified as a path (mip.tolerances.mipgap)
            # access one attribute at the time.
            if key == 'timelimit':
                # make sure we respect global timelimit
                value = min(value, seconds_left())
            attr = model.parameters
            for key_token in key.split('.'):
                attr = getattr(attr, key_token)
            attr.set(value)


class _CplexLoggerAdapter:
    """Wrap GALINI logger to work with cplex."""
    def __init__(self, logger, run_id, level):
        self._logger = logger
        self._run_id = run_id
        self._level = level

    def write(self, msg):
        """Write msg to logger."""
        self._logger.log(self._run_id, self._level, msg)

    def flush(self):
        """Flush logger."""
