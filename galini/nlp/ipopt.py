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
from pypopt import IpoptApplication, TNLP, NLPInfo
import numpy as np
from galini.core import HessianEvaluator
from galini.solvers import Solver


class GaliniTNLP(TNLP):
    def __init__(self, problem):
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

    def get_nlp_info(self):
        n = self.n
        m = self.m
        nnz_jac=n*m
        nnz_hess=(n*n + n)/2
        return NLPInfo(
            n=n,
            m=m,
            nnz_jac=nnz_jac,
            nnz_hess=nnz_hess,
        )

    def fill_bounds_info(self, x_l, x_u, g_l, g_u):
        # assert x_l.shape[0] == x_u.shape[0] == self.n
        # assert g_l.shape[0] == g_u.shape[0] == self.m

        # TODO: use correct infinity value
        for i, v in enumerate(self._problem.variables.values()):
            x_l[i] = v.lower_bound if v.lower_bound is not None else -2e19
            x_u[i] = v.upper_bound if v.upper_bound is not None else 2e19

        for i, c in enumerate(self._problem.constraints.values()):
            g_l[i] = c.lower_bound if c.lower_bound is not None else -2e19
            g_u[i] = c.upper_bound if c.upper_bound is not None else 2e19

        return True

    def fill_starting_point(self, init_x, x, init_z, z_l, z_u, init_lambda, lambda_):
        for i, v in enumerate(self._problem.variables.values()):
            if v.has_starting_point:
                x[i] = v.starting_point
            else:
                l = v.lower_bound if v.lower_bound is not None else -2e19
                u = v.upper_bound if v.upper_bound is not None else 2e19
                x[i] = max(l, min(u, 0))
        return True

    def fill_jacobian_g_structure(self, row, col):
        # TODO: real (sparse) structure
        for j in range(self.m):
            for i in range(self.n):
                row[j*self.n+i] = j
                col[j*self.n+i] = i

        return True

    def fill_hessian_structure(self, row, col):
        # TODO: real (sparse) structure
        idx = 0
        for i in range(self.n):
            for j in range(i+1):
                row[idx] = i
                col[idx] = j
                idx += 1

        return True

    def eval_f(self, x, new_x):
        self._ad.eval_at_x(x, new_x)
        return self._ad.values[self.objective_idx]

    def eval_grad_f(self, x, new_x, grad_f):
        self._ad.eval_at_x(x, new_x)
        grad = self._ad.jacobian[0, :]
        for i in range(self.n):
            grad_f[i] = grad[i]
        return True

    def eval_g(self, x, new_x, g):
        self._ad.eval_at_x(x, new_x)
        values = self._ad.values
        for i in range(self.m):
            g[i] = values[self.constraints_idx[i]]
        return True

    def eval_jacobian_g(self, x, new_x, jacobian):
        self._ad.eval_at_x(x, new_x)
        jac = self._ad.jacobian
        i = 0
        for r in range(self.m):
            for c in range(self.n):
                jacobian[i] = jac[r+1, c]
                i += 1
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
        return True

    def finalize_solution(self, x, z_l, z_u, g, lambda_, obj_value):
        # print(np.array(x, dtype=np.float64))
        pass


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
