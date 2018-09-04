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
import numpy as np


class IntervalHessianEvaluator(object):
    def __init__(self, problem):
        self.num_var = problem.num_variables
        self.num_cons = problem.num_constraints
        self.num_objs = problem.num_objectives
        self.vertices = problem.vertices
        self.size = problem.size

        self.values = np.empty(problem.size, dtype=object)
        self.dot = np.empty((problem.size, self.num_var), dtype=object)
        self.adj = np.empty(problem.size, dtype=object)
        self.adj_dot = np.empty((problem.size, self.num_var), dtype=object)

        self.jacobian = np.empty((self.num_cons + self.num_objs, self.num_var), dtype=object)
        self.hessian = np.empty((self.num_cons + self.num_objs, self.num_var, self.num_var), dtype=object)

        # precompute indexes of output nodes (objectives and constraints)
        self.output_idx = np.zeros(self.num_cons + self.num_objs, dtype=np.uint32)
        for i in range(self.num_objs):
            obj = problem._objectives[i]
            self.output_idx[i] = obj.root_expr.idx

        for i in range(self.num_cons):
            cons = problem._constraints[i]
            self.output_idx[self.num_objs+i] = cons.root_expr.idx

    def eval_at_x(self, x, new_x=True):
        if not new_x:
            return

        self._init_x(x)

        n_x = self.num_var
        # compute values once
        for i in range(n_x, self.size):
            expr = self.vertices[i]
            self.values[i] = expr.eval(self.values)

        # compute forward tangents
        for i in range(n_x, self.size):
            expr = self.vertices[i]
            for c in range(expr.num_children):
                j = expr.nth_children(c)
                d_v = expr.d_v(c, self.values)
                for k in range(n_x):
                    self.dot[i, k] += d_v * self.dot[j, k]

        for i in range(self.num_cons + self.num_objs):
            current = self.output_idx[i]
            self._compute_hessian(current)
            self.jacobian[i, :] = self.adj[:n_x]
            self.hessian[i, :, :] = self.adj_dot[:n_x, :n_x]

    def _compute_hessian(self, current_idx):
        n_x = self.num_var

        # reset from previous iterations
        self.adj[:] = 0
        self.adj_dot[:, :] = 0

        self.adj[current_idx] = 1.0

        for i in range(current_idx, n_x-1, -1):
            expr = self.vertices[i]
            for c in range(expr.num_children):
                j = expr.nth_children(c)
                d_v = expr.d_v(c, self.values)
                self.adj[j] += self.adj[i] * d_v

                for z in range(n_x):
                    self.adj_dot[j, z] += self.adj_dot[i, z] * d_v

                for c2 in range(expr.num_children):
                    k = expr.nth_children(c2)
                    dd_vv = expr.dd_vv(c, c2, self.values)
                    for z in range(n_x):
                        self.adj_dot[j, z] += self.adj[i] * dd_vv * self.dot[k, z]

    def _init_x(self, x):
        n_x = len(x)
        if n_x != self.num_var:
            raise RuntimeError('input size mismatch')

        self.dot[:] = 0.0
        for i in range(n_x):
            self.dot[i, i] = 1.0
            self.values[i] = x[i]