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
"""P^OA(T) relaxation of P."""
from collections import namedtuple
import pulp
import numpy as np
from galini.core import Domain


OAProblem = namedtuple('OAProblem', ['lp', 'x', 'alpha'])


class MilpRelaxation(object):
    def relax(self, problem, **kwargs):
        # TODO(fra): how to handle unbounded problems?
        # TODO(fra): what if it's maximization problem?
        lp = pulp.LpProblem(problem.name + '_OA', pulp.LpMinimize)
        alpha = pulp.LpVariable('alpha')
        x = [self._pulp_variable(v) for v in problem.variables]

        lp += alpha
        relaxation = OAProblem(lp=lp, x=x, alpha=alpha)
        self.update_relaxation(problem, relaxation, **kwargs)
        return relaxation

    def update_relaxation(self, problem, relaxation, **kwargs):
        x_k = kwargs.pop('x_k', None)

        if x_k is None:
            raise ValueError('Missing required kwarg "x_k"')

        if len(x_k) != problem.num_variables:
            raise ValueError('"x_k" must have same size as problem variables')

        # obtain indexes of "output" expressions: objectives and constraints
        f_idx = [f.root_expr.idx for f in problem.objectives]
        g_idx = [g.root_expr.idx for g in problem.constraints]

        fg = problem.expression_tree_data().eval(x_k, f_idx + g_idx)

        # evaluate at x_k
        fg_x = fg.forward(0, x_k)
        num_objectives = problem.num_objectives
        num_constraints = problem.num_constraints
        assert len(fg_x) == num_objectives + num_constraints

        w = np.zeros(num_objectives + num_constraints)
        lp = relaxation.lp

        for i in range(num_objectives + num_constraints):
            # compute gradient of f_i(x)
            w[i] = 1.0
            d_fg = fg.reverse(1, w)
            w[i] = 0.0

            expr = np.dot(d_fg, relaxation.x - x_k) + fg_x[i]

            if i < num_objectives:
                lp += expr <= relaxation.alpha
            else:
                constraint = problem.constraints[i-num_objectives]
                if constraint.lower_bound is not None:
                    # f(x) >= lb -> -f(x) <= lb
                    lp += expr >= constraint.lower_bound
                if constraint.upper_bound is not None:
                    lp += expr <= constraint.upper_bound

    def _pulp_variable(self, variable):
        # TODO(fra): can continuous variable be unbounded?
        assert variable.lower_bound is not None
        assert variable.upper_bound is not None
        domain = pulp.LpContinuous
        if variable.domain == Domain.INTEGER:
            domain = pulp.LpInteger
        elif variable.domain == Domain.BINARY:
            domain = pulp.LpBinary
        return pulp.LpVariable(
            variable.name,
            variable.lower_bound,
            variable.upper_bound,
            domain,
        )
