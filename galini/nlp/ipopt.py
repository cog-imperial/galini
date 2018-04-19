from galini.ad import ProblemAutodiff
from pypopt import IpoptApplication, TNLP, NLPInfo
import numpy as np


class GaliniTNLP(TNLP):
    def __init__(self, problem):
        super().__init__()
        self._problem = problem
        self._variables = self._problem.variables.values()
        self._constraints = self._problem.constraints.values()
        self._objective = list(self._problem.objectives.values())[0]
        self._ad = ProblemAutodiff(problem)

    def get_nlp_info(self):
        n = len(self._variables)
        m = len(self._constraints)
        nnz_jac=n*m
        nnz_hess=(n*n + n)/2
        return NLPInfo(
            n=n,
            m=m,
            nnz_jac=nnz_jac,
            nnz_hess=nnz_hess,
        )

    def fill_bounds_info(self, x_l, x_u, g_l, g_u):
        assert x_l.shape[0] == x_u.shape[0] == len(self._variables)
        assert g_l.shape[0] == g_u.shape[0] == len(self._constraints)

        # TODO: use correct infinity value
        for i, v in enumerate(self._variables):
            x_l[i] = v.lower_bound if v.lower_bound is not None else -2e19
            x_u[i] = v.upper_bound if v.upper_bound is not None else 2e19

        for i, c in enumerate(self._constraints):
            g_l[i] = c.lower_bound if c.lower_bound is not None else -2e19
            g_u[i] = c.upper_bound if c.upper_bound is not None else 2e19

        return True

    def fill_starting_point(self, init_x, x, init_z, z_l, z_u, init_lambda, lambda_):
        # TODO: real starting point
        for i, v in enumerate(self._variables):
            l = v.lower_bound if v.lower_bound is not None else -2e19
            u = v.upper_bound if v.upper_bound is not None else 2e19
            x[i] = max(l, min(u, 0))
        return True

    def fill_jacobian_g_structure(self, row, col):
        # TODO: real (sparse) structure
        m = len(self._constraints)
        n = len(self._variables)
        for j in range(m):
            for i in range(n):
                row[j*n+i] = j
                col[j*n+i] = i

        return True

    def fill_hessian_structure(self, row, col):
        # TODO: real (sparse) structure
        n = len(self._variables)
        idx = 0
        for i in range(n):
            for j in range(i+1):
                row[idx] = i
                col[idx] = j
                idx += 1

        return True

    def eval_f(self, x, new_x):
        self._ad.eval_at_x(x, new_x)
        # expected =  x[0] * x[3] * (x[0] + x[1] + x[2]) + x[2]
        f = self._ad.vertex_value(self._objective)
        # assert np.isclose(f, expected)
        return f

    def eval_grad_f(self, x, new_x, grad_f):
        self._ad.eval_at_x(x, new_x)

        if False:
            expected = np.zeros(4)
            expected[0] = x[0] * x[3] + x[3] * (x[0] + x[1] + x[2]);
            expected[1] = x[0] * x[3];
            expected[2] = x[0] * x[3] + 1;
            expected[3] = x[0] * (x[0] + x[1] + x[2]);

        grad = self._ad._jac[self._objective]
        for i in range(len(self._variables)):
            grad_f[i] = grad[i]
        #print('grad_f({}) = {}'.format(np.array(x), np.array(grad_f)))
        return True

    def eval_g(self, x, new_x, g):
        self._ad.eval_at_x(x, new_x)

        if False:
            expected = np.zeros(2)
            expected[0] = x[0] * x[1] * x[2] * x[3];
            expected[1] = x[0]*x[0] + x[1]*x[1] + x[2]*x[2] + x[3]*x[3];

        for i, cons in enumerate(self._constraints):
            g[i] = self._ad.vertex_value(cons)

        # assert np.allclose(g, expected)
        return True

    def eval_jacobian_g(self, x, new_x, jacobian):
        self._ad.eval_at_x(x, new_x)
        if False:
            values = np.zeros(8)
            values[0] = x[1]*x[2]*x[3]
            values[1] = x[0]*x[2]*x[3]
            values[2] = x[0]*x[1]*x[3]
            values[3] = x[0]*x[1]*x[2]

            values[4] = 2*x[0]
            values[5] = 2*x[1]
            values[6] = 2*x[2]
            values[7] = 2*x[3]

        for i, constraint in enumerate(self._constraints):
            j = self._ad._jac[constraint]
            # print('c = {}'.format(constraint.name))
            # print(j)
            # print()
            jacobian[i*j.shape[0]:(i+1)*j.shape[0]] = j
        # assert np.allclose(jacobian, values)
        return True

    def eval_hessian(self, x, new_x, obj_factor, lambda_, new_lambda, hess):
        self._ad.eval_at_x(x, new_x)

        if False:
            values = np.zeros(10)

            values[0] = obj_factor * (2*x[3])
            values[1] = obj_factor * (x[3])
            values[2] = 0.0
            values[3] = obj_factor * (x[3])
            values[4] = 0.0
            values[5] = 0.0
            values[6] = obj_factor * (2*x[0] + x[1] + x[2])
            values[7] = obj_factor * (x[0])
            values[8] = obj_factor * (x[0])
            values[9] = 0.0


            # add the portion for the first constraint
            values[1] += lambda_[0] * (x[2] * x[3])
            values[3] += lambda_[0] * (x[1] * x[3])
            values[4] += lambda_[0] * (x[0] * x[3])
            values[6] += lambda_[0] * (x[1] * x[2])
            values[7] += lambda_[0] * (x[0] * x[2])
            values[8] += lambda_[0] * (x[0] * x[1])

            # add the portion for the second constraint
            values[0] += lambda_[1] * 2
            values[2] += lambda_[1] * 2
            values[5] += lambda_[1] * 2
            values[9] += lambda_[1] * 2

        idxs = np.tril_indices(len(self._variables))
        obj_hes = self._ad._hes[self._objective]
        # print('hessian({})'.format(np.array(x)))
        # print('obj = ')
        # print(obj_hes)
        # print()
        hess[:]  = obj_factor * obj_hes[idxs]

        for i, constraint in enumerate(self._constraints):
            cons_hes = self._ad._hes[constraint]
            #print('c {} = '.format(constraint.name))
            #print(cons_hes)
            #print()
            hess[:] += lambda_[i] * cons_hes[idxs]

        #print()
        # assert np.allclose(values, hess)
        return True


    def finalize_solution(self, x, z_l, z_u, g, lambda_, obj_value):
        print(np.array(x, dtype=np.float64))


class IpoptNLPSolver(object):
    def __init__(self, config):
        self.config = config.get_group('ipopt')
        self.app = IpoptApplication()
        self.app.initialize()

    def solve(self, problem):
        tnlp = GaliniTNLP(problem)
        self.app.optimize_tnlp(tnlp)
