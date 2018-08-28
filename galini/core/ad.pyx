# cython: cdivision=True, boundscheck=False, wraparound=False
cimport numpy as np
import numpy as np
from galini.core.expression import float_, index_
from galini.core.expression cimport (
    foo,
    float_t,
    index,
    Expression,
    Objective,
    Constraint,
)
from galini.core.problem cimport Problem

cdef class _JacobianEvaluator:
    cdef readonly np.ndarray values
    cdef readonly np.ndarray jacobian
    cdef index num_var, num_cons, num_objs, size
    cdef object vertices

    def __init__(self, Problem problem):
        self.num_var = problem.num_variables
        self.num_cons = problem.num_constraints
        self.num_objs = problem.num_objectives
        self.vertices = problem.vertices
        self.size = problem.size
        self.values = np.zeros(problem.size, dtype=float_)
        self.jacobian = np.zeros((self.num_cons + self.num_objs, self.num_var), dtype=float_)

    cdef _init_x(self, foo[:] x):
        cdef index n_x, i
        cdef foo[:] values = self.values

        n_x = len(x)
        if n_x != self.num_var:
            raise RuntimeError('input size mismatch: {} expected {}'.format(n_x, self.num_var))

        for i in range(n_x):
            self.values[i] = x[i]

    def eval_at_x(self, foo[:] x, bint new_x):
        pass


cdef class ForwardJacobianEvaluator(_JacobianEvaluator):
    cdef np.ndarray dot
    cdef np.ndarray output_idx

    def __init__(self, Problem problem):
        super().__init__(problem)
        cdef index i, num_cons, num_objs
        cdef Objective obj
        cdef Constraint cons

        num_cons = self.num_cons
        num_objs = self.num_objs

        # contains tangents of expressions at x
        self.dot = np.zeros((problem.size, self.num_var), dtype=float_)

        # precompute indexes of output nodes (objectives and constraints)
        self.output_idx = np.zeros(num_cons + num_objs, dtype=index_)
        for i in range(num_objs):
            obj = problem._objectives[i]
            self.output_idx[i] = obj.root_expr.idx
        for i in range(num_cons):
            cons = problem._constraints[i]
            self.output_idx[num_objs+i] = cons.root_expr.idx

    def eval_at_x(self, foo[:] x, bint new_x):
        cdef index i, j, k, j_idx
        cdef foo d_v
        cdef Expression expr
        cdef index n_x

        if not new_x:
            return

        self._init_x(x)

        n_x = self.num_var
        cdef foo[:] values = self.values
        cdef foo[:, :] dot = self.dot

        dot[:] = 0.0
        for i in range(n_x):
            dot[i, i] = 1.0

        for i in range(n_x, self.size):
            expr = self.vertices[i]
            values[i] = expr.eval(values)
            for j in range(expr.num_children):
                j_idx = expr._nth_children(j)
                d_v = expr.d_v(j, values)
                for k in range(n_x):
                    dot[i, k] = dot[i, k] + d_v * dot[j_idx, k]

        self.jacobian = self.dot[self.output_idx]


cdef class ReverseJacobianEvaluator(_JacobianEvaluator):
    cdef np.ndarray adj
    cdef object constraints
    cdef object objectives

    def __init__(self, Problem problem):
        super().__init__(problem)
        self.constraints = problem._constraints
        self.objectives = problem._objectives

        # contains the adjoints at x
        self.adj = np.zeros(problem.size, dtype=float_)

    def eval_at_x(self, foo[:] x, bint new_x):
        cdef index i, j, root_idx
        cdef index num_objs, num_cons, n_x
        cdef Expression expr
        cdef Objective obj
        cdef Constraint cons
        cdef foo[:] adj, values
        cdef foo[:, :] jacobian = self.jacobian

        if not new_x:
            return

        n_x = self.num_var
        num_objs = self.num_objs
        num_cons = self.num_cons

        adj = self.adj
        values = self.values

        self._init_x(x)

        for i in range(n_x, self.size):
            expr = self.vertices[i]
            values[i] = expr.eval(values)

        for i in range(num_objs):
            obj = self.objectives[i]
            root_idx = obj.root_expr.idx
            self._eval_jacobian_for_output(root_idx, n_x, values, adj)
            jacobian[i, :] = adj[:n_x]

        for i in range(num_cons):
            cons = self.constraints[i]
            root_idx = cons.root_expr.idx
            self._eval_jacobian_for_output(root_idx, n_x, values, adj)
            jacobian[i + num_objs, :] = adj[:n_x]

    cdef void _eval_jacobian_for_output(self, index root_idx, index n_x, foo[:] values, foo[:] adj):
        cdef index i, j, child_idx
        cdef Expression expr
        adj[:] = 0.0
        adj[root_idx] = 1.0
        for i in range(root_idx, n_x-1, -1):
            expr = self.vertices[i]
            if adj[i] == 0.0:
                continue
            for j in range(expr.num_children):
                child_idx = expr._nth_children(j)
                adj[child_idx] = adj[child_idx] + adj[i] * expr.d_v(j, values)


class JacobianEvaluator(object):
    def __new__(cls, Problem problem):
        if problem.num_variables > problem.num_constraints:
            return ReverseJacobianEvaluator(problem)
        else:
            return ForwardJacobianEvaluator(problem)


cdef class HessianEvaluator:
    cdef readonly np.ndarray values, jacobian, hessian
    cdef np.ndarray dot, adj, adj_dot
    cdef index num_var, num_cons, num_objs, size
    cdef np.ndarray output_idx
    cdef object vertices

    def __init__(self, Problem problem):
        cdef index i
        cdef Objective obj
        cdef Constraint cons

        self.num_var = problem.num_variables
        self.num_cons = problem.num_constraints
        self.num_objs = problem.num_objectives
        self.vertices = problem.vertices
        self.size = problem.size

        self.values = np.zeros(problem.size, dtype=float_)
        self.dot = np.zeros((problem.size, self.num_var), dtype=float_)
        self.adj = np.zeros(problem.size, dtype=float_)
        self.adj_dot = np.zeros((problem.size, self.num_var), dtype=float_)

        self.jacobian = np.zeros((self.num_cons + self.num_objs, self.num_var), dtype=float_)
        self.hessian = np.zeros((self.num_cons + self.num_objs, self.num_var, self.num_var), dtype=float_)

        # precompute indexes of output nodes (objectives and constraints)
        self.output_idx = np.zeros(self.num_cons + self.num_objs, dtype=index_)
        for i in range(self.num_objs):
            obj = problem._objectives[i]
            self.output_idx[i] = obj.root_expr.idx

        for i in range(self.num_cons):
            cons = problem._constraints[i]
            self.output_idx[self.num_objs+i] = cons.root_expr.idx

    def eval_at_x(self, foo[:] x, new_x=True):
        cdef bint new_x_
        if new_x:
            new_x_ = 1
        else:
            new_x_ = 0
        return self._eval_at_x(x, new_x_)

    cdef void _eval_at_x(self, foo[:] x, bint new_x):
        cdef index i, current, n_x
        cdef Expression expr
        cdef foo d_v
        cdef foo[:] values = self.values
        cdef foo[:] adj = self.adj
        cdef foo[:, :] dot = self.dot
        cdef foo[:, :] adj_dot = self.adj_dot
        cdef foo[:, :] jacobian = self.jacobian
        cdef foo[:, :, :] hessian = self.hessian

        cdef index num_cons = self.num_cons
        cdef index num_objs = self.num_objs

        if not new_x:
            return

        self._init_x(x)

        n_x = self.num_var

        # compute values only once
        for i in range(n_x, self.size):
            expr = self.vertices[i]
            values[i] = expr.eval(values)

        # forward iteration, compute tangents
        for i in range(n_x, self.size):
            expr = self.vertices[i]
            for c in range(expr.num_children):
                j = expr._nth_children(c)
                d_v = expr.d_v(c, values)
                for k in range(n_x):
                    dot[i, k] = dot[i, k] + d_v * dot[j, k]

        for i in range(self.num_cons + self.num_objs):
            current = self.output_idx[i]
            self._compute_hessian(current)
            jacobian[i, :] = adj[:n_x]
            hessian[i, :, :] = adj_dot[:n_x, :n_x]

    cdef void _compute_hessian(self, index current):
        cdef index n_x, i, j, k, z, c, c2
        cdef Expression expr
        cdef foo d_v, dd_vv
        cdef foo[:] values = self.values
        cdef foo[:, :] dot = self.dot
        cdef foo[:] adj = self.adj
        cdef foo[:, :] adj_dot = self.adj_dot

        n_x = self.num_var

        # reset from previous iterations/calls
        adj[:] = 0.0
        adj_dot[:, :] = 0.0

        adj[current] = 1.0

        for i in range(current, n_x-1, -1):
            expr = self.vertices[i]
            for c in range(expr.num_children):
                j = expr._nth_children(c)
                d_v = expr.d_v(c, values)
                adj[j] = adj[j] + adj[i] * d_v

                for z in range(n_x):
                    adj_dot[j, z] = adj_dot[j, z] + adj_dot[i, z] * d_v

                for c2 in range(expr.num_children):
                    k = expr._nth_children(c2)
                    dd_vv = expr.dd_vv(c, c2, values)
                    for z in range(n_x):
                        adj_dot[j, z] = adj_dot[j, z] + adj[i] * dd_vv * dot[k, z]

    cdef _init_x(self, foo[:] x):
        cdef index n_x, i
        cdef foo[:] values = self.values
        cdef foo[:, :] dot = self.dot

        n_x = len(x)
        if n_x != self.num_var:
            raise RuntimeError('input size mismatch: {} expected {}'.format(n_x, self.num_var))

        dot[:] = 0.0
        for i in range(n_x):
            values[i] = x[i]
            dot[i, i] = 1.0
