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

"""Solve NLP using Ipopt."""
from typing import Optional
from pypopt import IpoptApplication, TNLP, NLPInfo
import numpy as np
import  galini.logging as log
from galini.core import Problem, HessianEvaluator
from galini.solvers import Solver


class GaliniTNLP(TNLP):
    """Implementation of the TNLP interface from pypopt for a Galini Problem."""

    # pylint: disable=invalid-name
    def __init__(self, problem: Problem) -> None:
        super().__init__()
        self.n = problem.num_variables
        self.m = problem.num_constraints
        self._problem = problem
        self._ad = HessianEvaluator(problem)

        objectives = list(problem.objectives.values())
        assert len(objectives) == 1
        objective = objectives[0]
        self.objective_idx = objective.root_expr.idx

        self.constraints_idx = np.zeros(self.m, dtype=np.int32)
        for i, constraint in enumerate(problem.constraints.values()):
            self.constraints_idx[i] = constraint.root_expr.idx

    def get_nlp_info(self) -> NLPInfo:
        n = self.n
        m = self.m
        nnz_jac = n*m
        nnz_hess = (n*n + n)/2
        return NLPInfo(
            n=n,
            m=m,
            nnz_jac=nnz_jac,
            nnz_hess=nnz_hess,
        )

    def fill_bounds_info(self, x_l: Optional[memoryview], x_u: Optional[memoryview],
                         g_l: Optional[memoryview], g_u: Optional[memoryview]) -> bool:

        # TODO: use correct infinity value
        if x_l is not None and x_u is not None:
            for i, v in enumerate(self._problem.variables.values()):
                x_l[i] = v.lower_bound if v.lower_bound is not None else -2e19
                x_u[i] = v.upper_bound if v.upper_bound is not None else 2e19

        log.matrix('ipopt/bounds/x_l', np.array(x_l))
        log.matrix('ipopt/bounds/x_u', np.array(x_u))

        if g_l is not None and g_u is not None:
            for i, c in enumerate(self._problem.constraints.values()):
                g_l[i] = c.lower_bound if c.lower_bound is not None else -2e19
                g_u[i] = c.upper_bound if c.upper_bound is not None else 2e19

        log.matrix('ipopt/bounds/g_l', np.array(g_l))
        log.matrix('ipopt/bounds/g_u', np.array(g_u))

        return True

    def fill_starting_point(self, init_x, x, init_z, z_l, z_u, init_lambda, lambda_):
        for i, v in enumerate(self._problem.variables.values()):
            if v.has_starting_point:
                x[i] = v.starting_point
            else:
                l = v.lower_bound if v.lower_bound is not None else -2e19
                u = v.upper_bound if v.upper_bound is not None else 2e19
                x[i] = max(l, min(u, 0))
        log.matrix('ipopt/starting_point', np.array(x))
        return True

    def fill_jacobian_g_structure(self, row, col):
        # TODO: real (sparse) structure
        for j in range(self.m):
            for i in range(self.n):
                row[j*self.n+i] = j
                col[j*self.n+i] = i

        log.matrix('ipopt/jacobian_g_structure/row', np.array(row))
        log.matrix('ipopt/jacobian_g_structure/col', np.array(col))
        return True

    def fill_hessian_structure(self, row, col):
        # TODO: real (sparse) structure
        idx = 0
        for i in range(self.n):
            for j in range(i+1):
                row[idx] = i
                col[idx] = j
                idx += 1

        log.matrix('ipopt/hessian_structure/row', np.array(row))
        log.matrix('ipopt/hessian_structure/col', np.array(col))
        return True

    def eval_f(self, x, new_x):
        self._ad.eval_at_x(x, new_x)
        return self._ad.values[self.objective_idx]

    def eval_grad_f(self, x, new_x, grad_f):
        self._ad.eval_at_x(x, new_x)
        grad = self._ad.jacobian[0, :]
        for i in range(self.n):
            grad_f[i] = grad[i]
        log.matrix('ipopt/grad_f', np.array(grad_f))
        return True

    def eval_g(self, x, new_x, g):
        self._ad.eval_at_x(x, new_x)
        values = self._ad.values
        for i in range(self.m):
            g[i] = values[self.constraints_idx[i]]
        log.matrix('ipopt/g', np.array(g))
        return True

    def eval_jacobian_g(self, x, new_x, jacobian):
        self._ad.eval_at_x(x, new_x)
        jac = self._ad.jacobian
        i = 0
        for r in range(self.m):
            for c in range(self.n):
                jacobian[i] = jac[r+1, c]
                i += 1
        log.matrix('ipopt/jacobian_x', np.array(x))
        log.matrix('ipopt/jacobian', np.array(jacobian))
        return True

    def eval_hessian(self, x, new_x, obj_factor, lambda_, new_lambda, hess):
        self._ad.eval_at_x(x, new_x)
        hessian = self._ad.hessian

        idx = 0
        for i in range(self.n):
            for j in range(i+1):
                hess[idx] = obj_factor * hessian[0, i, j]
                idx += 1

        for c in range(self.m):
            idx = 0
            for i in range(self.n):
                for j in range(i+1):
                    hess[idx] += lambda_[c] * hessian[c+1, i, j]
                    idx += 1
        log.matrix('ipopt/hessian_x', np.array(x))
        log.matrix('ipopt/hessian', np.array(hess))
        return True

    def finalize_solution(self, x, z_l, z_u, g, lambda_, obj_value):
        # print(np.array(x, dtype=np.float64))
        log.matrix('ipopt/solution/x', np.array(x))
        log.matrix('ipopt/solution/z_l', np.array(z_l))
        log.matrix('ipopt/solution/z_u', np.array(z_u))
        log.matrix('ipopt/solution/g', np.array(g))
        log.matrix('ipopt/solution/lambda', np.array(lambda_))
        log.matrix('ipopt/solution/obj_value', obj_value)


class IpoptNLPSolver(Solver):
    """Solver for NLP problems that uses Ipopt."""
    def __init__(self, config, mip_solver_registry, nlp_solver_registry):
        super().__init__(config, mip_solver_registry, nlp_solver_registry)
        self.config = config.get_group('ipopt')
        self.app = IpoptApplication()
        self.app.initialize()
        self._apply_config()

    def solve(self, problem, **kwargs):
        if problem.num_objectives != 1:
            raise RuntimeError('IpoptNLPSolver expects problems with 1 objective function.')
        tnlp = GaliniTNLP(problem)
        self.app.optimize_tnlp(tnlp)

    def _apply_config(self):
        options = self.app.options()
        for key, value in self.config.items():
            if isinstance(value, str):
                options.set_string_value(key, value)
            elif isinstance(value, int):
                options.set_integer_value(key, value)
            elif isinstance(value, float):
                options.set_numeric_value(key, value)
            else:
                raise RuntimeError('Invalid option type for {}'.format(key))
