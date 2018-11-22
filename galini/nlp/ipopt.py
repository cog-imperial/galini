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
import numpy as np
from pypopt import IpoptApplication, TNLP, NLPInfo
import  galini.logging as log
from galini.solvers import Solver, Solution, Status, OptimalObjective, OptimalVariable


class IpoptStatus(Status):
    DESCRIPTIONS = [
        'Optimal Solution Found',
    ]

    def __init__(self, status):
        self._status = status

    def is_success(self):
        return self._status == 0

    def description(self):
        return self.DESCRIPTIONS[self._status]


class DenseGaliniTNLP(TNLP):
    """Implementation of the TNLP interface from pypopt for a Galini Problem.

    Computes Jacobian and Hessian as dense matrices.
    """

    # pylint: disable=invalid-name
    def __init__(self, problem, solver, retape=True):
        self.retape = retape

        self._solver_name = solver.name
        self._run_id = solver.run_id

        self._solution = None
        self._problem = problem
        self._nf = problem.num_objectives
        self._ng = problem.num_constraints
        self._nfg = self._nf + self._ng
        self._nx = problem.num_variables
        self._expr_data = problem.expression_tree_data()

        self._nnz_jac = self._nx * self._ng
        self._nnz_hess = (self._nx * (self._nx + 1)) / 2

        self._f_idx = [f.root_expr.idx for f in problem.objectives]
        self._g_idx = [g.root_expr.idx for g in problem.constraints]
        self._fg_idx = self._f_idx + self._g_idx

        self._fg = None
        self._x0 = np.zeros(self._nx)
        self._fg0 = np.zeros(self._nfg)

        self._step_idx = 0

    def log_tensor(self, group, dataset, data):
        group_name = '{}/ipopt'.format(self._step_idx)
        if group is not None:
            group_name += '/' + str(group)
        log.tensor(self._solver_name, self._run_id, group_name, dataset, data)

    def get_nlp_info(self):
        return NLPInfo(
            n = self._nx,
            m = self._ng,
            nnz_jac=self._nnz_jac,
            nnz_hess=self._nnz_hess,
        )

    def get_bounds_info(self, x_l, x_u, g_l, g_u):
        if x_l is not None and x_u is not None:
            for i in range(self._nx):
                v = self._problem.variable_view(i)
                lb = v.lower_bound()
                ub = v.upper_bound()
                x_l[i] = lb if lb is not None else -2e19
                x_u[i] = ub if ub is not None else 2e19

        self.log_tensor('bounds', 'x_l', x_l)
        self.log_tensor('bounds', 'x_u', x_u)

        if g_l is not None and g_u is not None:
            for i in range(self._ng):
                c = self._problem.constraint(i)
                g_l[i] = c.lower_bound if c.lower_bound is not None else -2e19
                g_u[i] = c.upper_bound if c.upper_bound is not None else 2e19

        self.log_tensor('bounds', 'g_l', g_l)
        self.log_tensor('bounds', 'g_u', g_u)

        return True

    def get_starting_point(self, init_x, x, init_z, z_l, z_u, init_lambda, lambda_):
        for i in range(self._nx):
            v = self._problem.variable_view(i)
            if v.has_starting_point():
                x[i] = v.starting_point()
            else:
                lb = v.lower_bound()
                lb = lb if lb is not None else -2e19

                ub = v.upper_bound()
                ub = ub if ub is not None else 2e19

                x[i] = max(lb, min(ub, 0))

        self._cache_new_x(x, initial=True)

        self.log_tensor(None, 'starting_point', x)
        return True

    def get_jac_g_structure(self, row, col):
        for i in range(self._ng):
            for j in range(self._nx):
                idx = i * self._nx + j
                row[idx] = i
                col[idx] = j

        self.log_tensor('jac_g_structure', 'row', row)
        self.log_tensor('jac_g_structure', 'col', col)
        return True

    def get_h_structure(self, row, col):
        idx = 0
        for i in range(self._nx):
            for j in range(i+1):
                row[idx] = i
                col[idx] = j
                idx += 1

        self.log_tensor('hess_structure', 'row', row)
        self.log_tensor('hess_structure', 'col', col)
        return True

    def eval_f(self, x, new_x):
        if new_x:
            self._cache_new_x(x)

        # ipopt expects a scalar
        sum_ = np.sum(self._fg0[:self._nf])
        self.log_tensor(None, 'f', sum_)
        return sum_

    def eval_grad_f(self, x, new_x, grad_f):
        if new_x:
            self._cache_new_x(x)

        w = np.zeros(self._nfg)
        w[:self._nf] = 1.0
        grad = self._fg.reverse(1, w)
        for i in range(self._nx):
            grad_f[i] = grad[i]

        self.log_tensor(None, 'grad_f', grad_f)
        return True

    def eval_g(self, x, new_x, g):
        if new_x:
            self._cache_new_x(x)

        for i in range(self._ng):
            g[i] = self._fg0[self._nf + i]

        self.log_tensor(None, 'g', g)
        return True

    def eval_jac_g(self, x, new_x, jacobian):
        if new_x:
            self._cache_new_x(x)

        if False and self._nx < self._ng: # TODO(fra) implement fwd mode
            # use forward mode
            pass
        else:
            # use reverse mode
            w = np.zeros(self._nfg)
            for i in range(self._ng):
                w[self._nf + i] = 1.0
                jacobian_ = self._fg.reverse(1, w)
                for j in range(self._nx):
                    idx = i * self._nx + j
                    jacobian[idx] = jacobian_[j]
                w[self._nf + i] = 0.0
        self.log_tensor(None, 'jacobian', jacobian)
        return True

    def eval_h(self, x, new_x, obj_factor, lambda_, new_lambda, hess):
        if new_x:
            self._cache_new_x(x)

        w = np.zeros(self._nfg)
        for i in range(self._nf):
            w[i] = obj_factor
        for i in range(self._ng):
            w[i+self._nf] = lambda_[i]

        hess_ = self._fg.hessian(x, w)
        for i in range(len(hess)):
            hess[i] = hess_[i]
        return True

    def finalize_solution(self, status, x, z_l, z_u, g, lambda_, obj_value):
        status = IpoptStatus(status)
        if status.is_success():
            opt_objs = [OptimalObjective(obj.name, obj_value)
                        for obj in self._problem.objectives]
            opt_vars = [OptimalVariable(var.name, x[i])
                        for i, var in enumerate(self._problem.variables)]
            self._solution = Solution(status, opt_objs, opt_vars)

        self.log_tensor('solution', 'x', x)
        self.log_tensor('solution', 'z_l', z_l)
        self.log_tensor('solution', 'z_u', z_u)
        self.log_tensor('solution', 'g', g)
        self.log_tensor('solution', 'lambda', lambda_)
        self.log_tensor('solution', 'obj_value', obj_value)

    def _cache_new_x(self, x, initial=False):
        self._x0[:] = x
        if self.retape or initial:
            self._fg = self._expr_data.eval(self._x0, self._fg_idx)
        self._fg0[:] = self._fg.forward(0, self._x0)
        self._step_idx += 1


class IpoptNLPSolver(Solver):
    """Solver for NLP problems that uses Ipopt."""
    name = 'ipopt'
    def __init__(self, config, mip_solver_registry, nlp_solver_registry):
        super().__init__(config, mip_solver_registry, nlp_solver_registry)
        self.config = config.ipopt
        self.app = IpoptApplication()
        self.app.initialize()
        self._apply_config()

    def solve(self, problem, **kwargs):
        if problem.num_objectives != 1:
            raise RuntimeError('IpoptNLPSolver expects problems with 1 objective function.')
        if self.sparse:
            raise RuntimeError('Sparse IpoptTNLP not yet implemented.')
        else:
            tnlp = DenseGaliniTNLP(problem, solver=self)
        self.app.optimize_tnlp(tnlp)
        return tnlp._solution

    def _apply_config(self):
        # pop nonipopt options
        self.sparse = self.config.pop('sparse', False)

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
