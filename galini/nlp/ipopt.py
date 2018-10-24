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
from galini.solvers import Solver


class GaliniTNLP(TNLP):
    """Implementation of the TNLP interface from pypopt for a Galini Problem."""

    # pylint: disable=invalid-name
    def __init__(self, problem):
        pass

    def get_nlp_info(self):
        pass

    def get_bounds_info(self, x_l, x_u, g_l, g_u):
        return True

    def get_starting_point(self, init_x, x, init_z, z_l, z_u, init_lambda, lambda_):
        return True

    def get_jac_g_structure(self, row, col):
        return True

    def get_h_structure(self, row, col):
        return True

    def eval_f(self, x, new_x):
        pass

    def eval_grad_f(self, x, new_x, grad_f):
        return True

    def eval_g(self, x, new_x, g):
        return True

    def eval_jac_g(self, x, new_x, jacobian):
        return True

    def finalize_solution(self, _status, x, z_l, z_u, g, lambda_, obj_value):
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
        self.config = config.ipopt
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
