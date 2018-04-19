import math
import numpy as np
import galini.dag.expressions as dex


def eval_vertex(idx, vertex, i, v):
    if isinstance(vertex, dex.Constant):
        return vertex.value
    elif isinstance(vertex, dex.ProductExpression):
        a, b = vertex.children
        return v[idx[a]] * v[idx[b]]
    elif isinstance(vertex, dex.DivisionExpression):
        a, b = vertex.children
        if np.isclose(v[idx[b]], 0.0):
            return np.sign(v[idx[a]]) * np.sign(v[idx[b]]) * 2e19
        return v[idx[a]] / v[idx[b]]
    elif isinstance(vertex, dex.PowExpression):
        base, expo = vertex.children
        return v[idx[base]]**v[idx[expo]]
    elif isinstance(vertex, dex.LinearExpression):
        values = [v[idx[ch]] for ch in vertex.children]
        return sum(
            c * v
            for c, v in zip(vertex.coefficients, values)
        ) + vertex.constant_term
    elif isinstance(vertex, dex.SumExpression):
        return sum(v[idx[ch]] for ch in vertex.children)
    elif isinstance(vertex, (dex.Constraint, dex.Objective)):
        return v[idx[vertex.children[0]]]
    elif isinstance(vertex, dex.NegationExpression):
        return -v[idx[vertex.children[0]]]
    elif isinstance(vertex, dex.AbsExpression):
        return abs(v[idx[vertex.children[0]]])
    print(vertex)
    assert False


def eval_d_v(idx, vertex, i, j, v):
    if isinstance(vertex, dex.Constant):
        raise RuntimeError('trying to differentiate constant')
    elif isinstance(vertex, dex.ProductExpression):
        a, b = vertex.children
        a_i = idx[a]
        b_i = idx[b]
        if j == a_i:
            return v[b_i]
        else:
            return v[a_i]

    elif isinstance(vertex, dex.DivisionExpression):
        a, b = vertex.children
        a_i = idx[a]
        b_i = idx[b]
        if j == a_i:
            if np.isclose(v[b_i], 0.0):
                return np.sign(v[b_i]) * 2e19
            return 1.0 / v[b_i]
        else:
            if np.isclose(v[b_i], 0.0):
                return -1 * np.sign(v[a_i]) * 2e19
            return -v[a_i] / (v[b_i]**2.0)

    elif isinstance(vertex, dex.PowExpression):
        base, expo = vertex.children
        v_base = v[idx[base]]
        v_expo = v[idx[expo]]
        if idx[base] == j:
            if v_expo == 1.0:
                return 1.0
            elif v_expo == 2.0:
                return 2*v_base
            elif v_expo == int(v_expo):
                return v_expo * (v_base**(v_expo - 1))
            else:
                return v_base**v_expo * (v_expo / v_base)
        else:
            if np.isclose(v_base, 0):
                return -2e19
            return v_base**v_expo * math.log(v_base)
    elif isinstance(vertex, dex.LinearExpression):
        # need to multiply with coefficient
        for coef, ch in zip(vertex.coefficients, vertex.children):
            if idx[ch] == j:
                return coef
        raise RuntimeError('unreachable')
    elif isinstance(vertex, dex.SumExpression):
        return 1.0
    elif isinstance(vertex, (dex.Constraint, dex.Objective)):
        return v[idx[vertex.children[0]]]
    elif isinstance(vertex, dex.NegationExpression):
        return -1.0
    elif isinstance(vertex, dex.AbsExpression):
        if v[idx[vertex.children[0]]] > 0:
            return 1.0
        else:
            return -1.0
    print(vertex)
    assert False


def eval_dd_vv(idx, vertex, i, j, k, v):
    if isinstance(vertex, dex.Constant):
        raise RuntimeError('trying to differentiate constant')
    elif isinstance(vertex, dex.ProductExpression):
        a, b = vertex.children
        if j == k:
            if a is b:
                return 1.0
            else:
                return 0.0
        else:
            return 1.0

    elif isinstance(vertex, dex.DivisionExpression):
        a, b = vertex.children
        a_i = idx[a]
        b_i = idx[b]
        assert a_i != b_i
        if a_i == j:
            if j == k:
                return 0.0
            if np.isclose(v[b_i], 0.0):
                return -2e19
            return -1/(v[b_i]**2)
        else:
            if j == k:
                if np.isclose(v[b_i], 0.0):
                    return np.sign(v[b_i]) * 2e19
                return v[a_i]/(v[b_i]**3)

            if np.isclose(v[b_i], 0.0):
                return -2e19
            return -1/(v[b_i]**2)

    elif isinstance(vertex, dex.PowExpression):
        base, expo = vertex.children
        v_base = v[idx[base]]
        v_expo = v[idx[expo]]
        if idx[base] == j:
            if v_expo == 1.0:
                return 0.0
            elif v_expo == 2.0:
                return 2.0
            elif v_expo == int(v_expo):
                return (v_expo * (v_expo - 1))*(v_base**(v_expo - 2))
            else:
                # TODO
                return v_base**v_expo * (v_expo / v_base)
        else:
            if np.isclose(v_base, 0):
                return -2e19
            return v_base**v_expo * math.log(v_base)
    elif isinstance(vertex, dex.LinearExpression):
        return 0.0
    elif isinstance(vertex, dex.SumExpression):
        return 0.0
    elif isinstance(vertex, (dex.Constraint, dex.Objective)):
        return v[idx[vertex.children[0]]]
    elif isinstance(vertex, dex.NegationExpression):
        return 0.0
    elif isinstance(vertex, dex.AbsExpression):
        return 0.0
    print(vertex)
    assert False


class ProblemAutodiff(object):
    """Compute Jacobian and Hessian of the problem with automatic differentiation.

    Call `eval_at_x` to evaluate the functions and compute the problem Jacobian and
    Hessian at point `x`.


    References
    ----------

    [1] Walther, A. (2008). Computing sparse Hessians with automatic differentiation.
        ACM Transactions on Mathematical Software, 34(1), 1–15.
        https://doi.org/10.1145/1322436.1322439
    """
    def __init__(self, problem):
        self._problem = problem
        self._variables = problem.variables.values()
        self.n = len(self._variables)
        self._constraints = list(problem.constraints.values()) + list(problem.objectives.values())
        self.m = len(self._constraints)

        self.num_nodes = len(problem._vertices)

        self._v = np.zeros(self.num_nodes)
        self._V_dot = np.zeros((self.num_nodes, self.n))
        self._v_bar = np.zeros(self.num_nodes)
        self._V_bar_dot = np.zeros((self.num_nodes, self.n))

        self._jac = {}
        self._hes = {}

        self._idx = {}
        for i, vertex in enumerate(self._problem.vertices):
            self._idx[vertex] = i

    def vertex_value(self, vertex):
        return self._v[self._idx[vertex]]

    def eval_at_x(self, x, new_x=True):
        if not new_x:
            return

        self._v_bar[:] = 0.0
        self._v[:] = 0.0
        self._V_dot[:, :] = 0.0
        self._V_bar_dot[:, :] = 0.0

        for i in range(self.n):
            self._v[i] = x[i]
            self._V_dot[i, i] = 1.0

        for vertex in self._problem.vertices:
            i = self._idx[vertex]
            # skip variables
            if i < self.n:
                continue

            self._v[i] = eval_vertex(self._idx, vertex, i, self._v)
            if isinstance(vertex, (dex.Constraint, dex.Objective)):
                j = self._idx[vertex.children[0]]
                self._V_dot[i, :] = self._V_dot[j]
            else:
                for child in vertex.children:
                    j = self._idx[child]
                    self._V_dot[i, :] += eval_d_v(self._idx, vertex, i, j, self._v) * self._V_dot[j, :]

        # repeat for each constraint
        for constraint in self._constraints:
            self._v_bar[self._idx[constraint]] = 1.0
            self._v_bar[self._idx[constraint.children[0]]] = 1.0

            for vertex in reversed(list(self._problem.vertices)):
                if isinstance(vertex, (dex.Variable, dex.Constraint, dex.Objective)):
                    continue

                i = self._idx[vertex]
                for child in vertex.children:
                    j = self._idx[child]
                    self._v_bar[j] += self._v_bar[i] * eval_d_v(self._idx, vertex, i, j, self._v)
                    for child2 in vertex.children:
                        k = self._idx[child2]
                        self._V_bar_dot[j, :] += self._v_bar[i] * eval_dd_vv(self._idx, vertex, i, j, k, self._v) * self._V_dot[k, :]
                    self._V_bar_dot[j, :] += self._V_bar_dot[i, :] * eval_d_v(self._idx, vertex, i, j, self._v)

            self._jac[constraint] = np.copy(self._v_bar[:self.n])
            self._hes[constraint] = np.copy(self._V_bar_dot[:self.n, :self.n])
            self._v_bar[:] = 0.0
            self._V_bar_dot[:, :] = 0.0
