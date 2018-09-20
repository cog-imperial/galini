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

# cython: cdivision=True, boundscheck=False, wraparound=False
from cpython.mem cimport PyMem_Malloc, PyMem_Realloc, PyMem_Free
cimport libc.math as math
cimport numpy as np
import numpy as np
from suspect.expression import ExpressionType, UnaryFunctionType

float_ = np.float64
index_ = np.uint32

INFINITY = 2e19
cdef float_t C_INFINITY = 2e19


cdef class ExpressionChildren(object):
    """Expose list-like object containing an expression children."""
    cdef index[:] _children
    cdef object _problem

    def __init__(self, object problem, index[:] children):
        self._problem = problem
        self._children = children

    cdef object _nth_children(self, index i):
        cdef index idx = self._children[i]
        return self._problem.vertices[idx]

    def __bool__(self):
        return len(self._children) != 0

    def __getitem__(self, index item):
        return self._nth_children(item)

    def __len__(self):
        return len(self._children)

    def __iter__(self):
        cdef index i
        cdef index idx
        for i in range(len(self._children)):
            yield self._nth_children(i)


cdef class Expression:
    def __init__(self, int expression_type, object problem=None):
        self.problem = problem
        self.num_children = 0
        self.idx = 0
        # always come after variables (0) and constants(1)
        self.default_depth = 2
        self.expression_type = expression_type

    cdef void reindex(self, index cutoff) nogil:
        if self.idx >= cutoff:
            self.idx += 1

    cpdef bint is_constant(self):
        return 0

    def eval(self, object[:] v):
        return 0.0

    cdef float_t _eval(self, float_t[:] v):
        return 0.0

    def d_v(self, index j, object[:] v):
        return 0.0

    cdef float_t _d_v(self, index j, float_t[:] v):
        return 0.0

    def dd_vv(self, index j, index k, object[:] v):
        return 0.0

    cdef float_t _dd_vv(self, index j, index k, float_t[:] v):
        return 0.0

    cpdef index nth_children(self, index i):
        return self._nth_children(i)

    cdef index _nth_children(self, index i) nogil:
        return 0

    @property
    def children(self):
        return None


cdef class UnaryExpression(Expression):
    def __init__(self, object children, int expression_type, object problem=None):
        super().__init__(expression_type, problem)
        assert len(children) == 1
        self.num_children = 1
        self._children[0] = children[0]

    cdef void reindex(self, index cutoff) nogil:
        cdef index children_idx
        if self.idx >= cutoff:
            self.idx += 1

        children_idx = self._children[0]
        if children_idx >= cutoff:
            self._children[0] = children_idx + 1

    cdef index _nth_children(self, index i) nogil:
        return self._children[0]

    @property
    def children(self):
        cdef index[:] buff = <index[:1]>self._children
        if self.problem is None:
            raise RuntimeError('Expression must be part of problem')
        return ExpressionChildren(self.problem, buff)


cdef class BinaryExpression(Expression):
    def __init__(self, object children, int expression_type, object problem=None):
        super().__init__(expression_type, problem)
        assert len(children) == 2
        self.num_children = 2
        self._children[0] = children[0]
        self._children[1] = children[1]

    cdef void reindex(self, index cutoff) nogil:
        cdef index children_idx
        if self.idx >= cutoff:
            self.idx += 1

        for i in range(2):
            children_idx = self._children[i]
            if children_idx >= cutoff:
                self._children[i] = children_idx + 1

    cdef index _nth_children(self, index i) nogil:
        return self._children[i]

    @property
    def children(self):
        cdef index[:] buff = <index[:2]>self._children
        if self.problem is None:
            raise RuntimeError('Expression must be part of problem')
        return ExpressionChildren(self.problem, buff)


cdef class NaryExpression(Expression):
    def __init__(self, object children, int expression_type, object problem=None):
        super().__init__(expression_type, problem)

        cdef index num_children = len(children)
        assert num_children > 0
        self._children = <index *>PyMem_Malloc(num_children * sizeof(index))
        self.num_children = num_children

        cdef index i
        for i in range(num_children):
            self._children[i] = children[i]

    def __dealloc__(self):
        PyMem_Free(self._children)

    cdef void reindex(self, index cutoff) nogil:
        cdef index children_idx
        if self.idx >= cutoff:
            self.idx += 1

        for i in range(self.num_children):
            children_idx = self._children[i]
            if children_idx >= cutoff:
                self._children[i] = children_idx + 1

    cdef index _nth_children(self, index i) nogil:
        return self._children[i]

    @property
    def children(self):
        cdef index[:] buff = <index[:self.num_children]>self._children
        if self.problem is None:
            raise RuntimeError('Expression must be part of problem')
        return ExpressionChildren(self.problem, buff)


cdef class ProductExpression(BinaryExpression):
    def __init__(self, object children, object problem=None):
        super().__init__(children, ExpressionType.Product, problem)

    def eval(self, np.ndarray v):
        return v[self._children[0]] * v[self._children[1]]

    cdef float_t _eval(self, float_t[:] v):
        return v[self._children[0]] * v[self._children[1]]

    def d_v(self, index j, np.ndarray v):
        if j == 0:
            return v[self._children[1]]
        elif j == 1:
            return v[self._children[0]]
        return 0

    cdef float_t _d_v(self, index j, float_t[:] v):
        if j == 0:
            return v[self._children[1]]
        elif j == 1:
            return v[self._children[0]]
        return 0

    def dd_vv(self, index j, index k, np.ndarray v):
        if j == k:
            return 0.0
        else:
            return 1.0

    cdef float_t _dd_vv(self, index j, index k, float_t[:] v):
        if j == k:
            return 0.0
        else:
            return 1.0


cdef class DivisionExpression(BinaryExpression):
    def __init__(self, object children, object problem=None):
        super().__init__(children, ExpressionType.Division, problem)

    def eval(self, np.ndarray v):
        return v[self._children[0]] / v[self._children[1]]

    cdef float_t _eval(self, float_t[:] v):
        return v[self._children[0]] / v[self._children[1]]

    def d_v(self, index j, np.ndarray v):
        if j == 0:
            # d/dx (x/y) := 1/y
            return 1.0/v[self._children[1]]
        elif j == 1:
            # d/dy (x/y) := -x/y^2
            y = v[self._children[1]]
            return -v[self._children[0]] / (y*y)
        else:
            return 0

    cdef float_t _d_v(self, index j, float_t[:] v):
        cdef float_t y
        if j == 0:
            # d/dx (x/y) := 1/y
            return 1.0/v[self._children[1]]
        elif j == 1:
            # d/dy (x/y) := -x/y^2
            y = v[self._children[1]]
            return -v[self._children[0]] / (y*y)
        else:
            return 0

    def dd_vv(self, index j, index k, np.ndarray v):
        y = v[self._children[1]]
        if j == k:
            if j == 0:
                # d/dx^2
                return 0.0
            else:
                x = v[self._children[0]]
                return 2*x / (y*y*y)
        else:
            # d/dxdy == d/dydx == -1/y^2
            return -1.0 / (y*y)

    cdef float_t _dd_vv(self, index j, index k, float_t[:] v):
        cdef float_t x
        cdef float_t y = v[self._children[1]]
        if j == k:
            if j == 0:
                # d/dx^2
                return 0.0
            else:
                x = v[self._children[0]]
                return 2*x / (y*y*y)
        else:
            # d/dxdy == d/dydx == -1/y^2
            return -1.0 / (y*y)


cdef class SumExpression(NaryExpression):
    def __init__(self, object children, object problem=None):
        super().__init__(children, ExpressionType.Sum, problem)

    def eval(self, np.ndarray v):
        tot = 0
        for i in range(self.num_children):
            tot += v[self._children[i]]
        return tot

    cdef float_t _eval(self, float_t[:] v):
        cdef index i
        cdef float_t tot = 0
        for i in range(self.num_children):
            tot += v[self._children[i]]
        return tot

    def d_v(self, index j, np.ndarray v):
        return 1.0

    cdef float_t _d_v(self, index j, float_t[:] v):
        return 1.0

    def dd_vv(self, index j, index k, np.ndarray v):
        return 0.0

    cdef float_t _dd_vv(self, index j, index k, float_t[:] v):
        return 0.0


cdef class PowExpression(BinaryExpression):
    def __init__(self, object children, object problem=None):
        super().__init__(children, ExpressionType.Power, problem)

    def eval(self, np.ndarray v):
        base = v[self._children[0]]
        expo = v[self._children[1]]
        if expo == 0:
            return 0.0
        elif expo == 1:
            return base
        elif expo == 2:
            return base*base
        else:
            return base ** expo

    cdef float_t _eval(self, float_t[:] v):
        cdef float_t base = v[self._children[0]]
        cdef float_t expo = v[self._children[1]]
        if expo == 0:
            return 0.0
        elif expo == 1:
            return base
        elif expo == 2:
            return base*base
        else:
            return math.pow(base, expo)

    def d_v(self, index j, np.ndarray v):
        base = v[self._children[0]]
        expo = v[self._children[1]]
        if j == 0:
            # derive over base
            if expo == 1.0:
                return 1.0
            elif expo == 2.0:
                return 2*base
                #elif ceilf(expo) == expo:
                #    return expo * math.pow(base, expo-1)
            else:
                return (base ** expo) * (expo / base)
        elif j == 1:
            # derive over exponent
            if isinstance(expo, float):
                return 0
            if base.is_negative():
                return -INFINITY
            return (base ** expo) * base.log()
        else:
            return INFINITY

    cdef float_t _d_v(self, index j, float_t[:] v):
        cdef float_t base = v[self._children[0]]
        cdef float_t expo = v[self._children[1]]
        if j == 0:
            # derive over base
            if expo == 1.0:
                return 1.0
            elif expo == 2.0:
                return 2*base
                #elif ceilf(expo) == expo:
                #    return expo * math.pow(base, expo-1)
            else:
                return math.pow(base, expo) * (expo / base)
        elif j == 1:
            # derive over exponent
            if base < 1e-8:
                return -INFINITY
            return math.pow(base, expo) * math.log(base)
        else:
            return INFINITY

    def dd_vv(self, index j, index k, np.ndarray v):
        base = v[self._children[0]]
        expo = v[self._children[1]]
        if j != k:
            # derivative^2 over base,expo
            a = base ** expo
            l = base.log()
            b = expo / base
            c = a / base
            return a*l*b + c
        else:  # j == k
            if k == 0:
                # derivative^2 over base^2
                a = base ** expo
                b = expo / base
                c = expo / (base*base)
                return a*(b*b - c)
            else:
                # derivative^2 over expo^2
                a = base ** expo
                l = base.log()
                return a * l * l

    cdef float_t _dd_vv(self, index j, index k, float_t[:] v):
        cdef float_t a, l, b, c
        cdef float_t base = v[self._children[0]]
        cdef float_t expo = v[self._children[1]]
        if j != k:
            # derivative^2 over base,expo
            a = math.pow(base, expo)
            l = math.log(base)
            b = expo / base
            c = a / base
            return a*l*b + c
        else:  # j == k
            if k == 0:
                # derivative^2 over base^2
                a = math.pow(base, expo)
                b = expo / base
                c = expo / (base*base)
                return a*(b*b - c)
            else:
                # derivative^2 over expo^2
                a = math.pow(base, expo)
                l = math.log(base)
                return a * l * l


cdef class LinearExpression(NaryExpression):
    def __init__(self, object children, float_t[:] coefficients, float_t constant, object problem=None):
        super().__init__(children, ExpressionType.Linear, problem)

        self.constant = constant
        self._coefficients = <float_t *>PyMem_Malloc(self.num_children * sizeof(float_t))
        cdef float_t[:] coefficients_view = <float_t[:self.num_children]>self._coefficients
        assert self.num_children == len(coefficients)
        coefficients_view[:] = coefficients

    def __dealloc__(self):
        PyMem_Free(self._coefficients)

    @property
    def coefficients(self):
        return <float_t[:self.num_children]>self._coefficients

    def eval(self, np.ndarray v):
        tot = self.constant
        for i in range(self.num_children):
            tot += self._coefficients[i] * v[self._children[i]]
        return tot

    cdef float_t _eval(self, float_t[:] v):
        cdef index i
        cdef float_t tot = self.constant
        for i in range(self.num_children):
            tot += self._coefficients[i] * v[self._children[i]]
        return tot

    def d_v(self, index j, np.ndarray v):
        return self._coefficients[j]

    cdef float_t _d_v(self, index j, float_t[:] v):
        return self._coefficients[j]

    def dd_vv(self, index j, index k, np.ndarray v):
        return 0.0

    cdef float_t _dd_vv(self, index j, index k, float_t[:] v):
        return 0.0


cdef class NegationExpression(UnaryExpression):
    def __init__(self, object children):
        super().__init__(children, ExpressionType.Negation)

    def eval(self, np.ndarray v):
        return -v[self._children[0]]

    cdef float_t _eval(self, float_t[:] v):
        return -v[self._children[0]]

    def d_v(self, index j, np.ndarray v):
        return -1.0

    cdef float_t _d_v(self, index j, float_t[:] v):
        return -1.0


cdef class UnaryFunctionExpression(UnaryExpression):
    def __init__(self, object children, int func_type, object problem=None):
        super().__init__(children, ExpressionType.UnaryFunction, problem)
        self.func_type = func_type


cdef class AbsExpression(UnaryFunctionExpression):
    def __init__(self, object children, object problem=None):
        super().__init__(children, UnaryFunctionType.Abs, problem)

    def eval(self, np.ndarray v):
        x = v[self._children[0]]
        if x < 0:
            return -x
        else:
            return x

    cdef float_t _eval(self, float_t[:] v):
        cdef float_t x = v[self._children[0]]
        if x >= 0:
            return x
        else:
            return -x

    def d_v(self, index j, np.ndarray v):
        x = v[self._children[0]]
        if x < 0:
            return -1.0
        else:
            return 1.0

    cdef float_t _d_v(self, index j, float_t[:] v):
        cdef float_t x = v[self._children[0]]
        if x >= 0:
            return 1.0
        else:
            return -1.0


cdef class SqrtExpression(UnaryFunctionExpression):
    def __init__(self, object children, object problem=None):
        super().__init__(children, UnaryFunctionType.Sqrt, problem)

    def eval(self, np.ndarray v):
        return v[self._children[0]].sqrt()

    cdef float_t _eval(self, float_t[:] v):
        return math.sqrt(v[self._children[0]])

    def d_v(self, index j, np.ndarray v):
        a = v[self._children[0]]
        return 1.0 / (2.0 * a.sqrt())

    cdef float_t _d_v(self, index j, float_t[:] v):
        return 1.0 / (2.0 * math.sqrt(v[self._children[0]]))

    def dd_vv(self, index j, index k, np.ndarray v):
        a = v[self._children[0]]
        s = a.pow()
        return -1.0/(4.0 * (s ** 3))

    cdef float_t _dd_vv(self, index j, index k, float_t[:] v):
        cdef float_t s = math.sqrt(v[self._children[0]])
        return -1.0/(4.0 * math.pow(s, 3))


cdef class ExpExpression(UnaryFunctionExpression):
    def __init__(self, object children, object problem=None):
        super().__init__(children, UnaryFunctionType.Exp, problem)

    def eval(self, np.ndarray v):
        return v[self._children[0]].exp()

    cdef float_t _eval(self, float_t[:] v):
        return math.exp(v[self._children[0]])

    def d_v(self, index j, np.ndarray v):
        return v[self._children[0]].exp()

    cdef float_t _d_v(self, index j, float_t[:] v):
        return math.exp(v[self._children[0]])

    def dd_vv(self, index j, index k, np.ndarray v):
        return v[self._children[0]].exp()

    cdef float_t _dd_vv(self, index j, index k, float_t[:] v):
        return math.exp(v[self._children[0]])


cdef class LogExpression(UnaryFunctionExpression):
    def __init__(self, object children, object problem=None):
        super().__init__(children, UnaryFunctionType.Log, problem)

    def eval(self, np.ndarray v):
        return v[self._children[0]].log()

    cdef float_t _eval(self, float_t[:] v):
        return math.log(v[self._children[0]])

    def d_v(self, index j, np.ndarray v):
        return 1.0 / v[self._children[0]]

    cdef float_t _d_v(self, index j, float_t[:] v):
        return 1.0 / v[self._children[0]]

    def dd_vv(self, index j, index k, np.ndarray v):
        x = v[self._children[0]]
        return -1.0 / (x*x)

    cdef float_t _dd_vv(self, index j, index k, float_t[:] v):
        cdef float_t x = v[self._children[0]]
        return -1.0 / (x*x)


cdef class SinExpression(UnaryFunctionExpression):
    def __init__(self, object children, object problem=None):
        super().__init__(children, UnaryFunctionType.Sin, problem)

    def eval(self, np.ndarray v):
        return v[self._children[0]].sin()

    cdef float_t _eval(self, float_t[:] v):
        return math.sin(v[self._children[0]])

    def d_v(self, index j, np.ndarray v):
        return v[self._children[0]].cos()

    cdef float_t _d_v(self, index j, float_t[:] v):
        return math.cos(v[self._children[0]])

    def dd_vv(self, index j, index k, np.ndarray v):
        return -v[self._children[0]].sin()

    cdef float_t _dd_vv(self, index j, index k, float_t[:] v):
        return -math.sin(v[self._children[0]])


cdef class CosExpression(UnaryFunctionExpression):
    def __init__(self, object children, object problem=None):
        super().__init__(children, UnaryFunctionType.Cos, problem)

    def eval(self, np.ndarray v):
        return v[self._children[0]].cos()

    cdef float_t _eval(self, float_t[:] v):
        return math.cos(v[self._children[0]])

    def d_v(self, index j, np.ndarray v):
        return -v[self._children[0]].sin()

    cdef float_t _d_v(self, index j, float_t[:] v):
        return -math.sin(v[self._children[0]])

    def dd_vv(self, index j, index k, np.ndarray v):
        return -v[self._children[0]].cos()

    cdef float_t _dd_vv(self, index j, index k, float_t[:] v):
        return -math.cos(v[self._children[0]])


cdef class TanExpression(UnaryFunctionExpression):
    def __init__(self, object children, object problem=None):
        super().__init__(children, UnaryFunctionType.Tan, problem)

    def eval(self, np.ndarray v):
        return v[self._children[0]].tan()

    cdef float_t _eval(self, float_t[:] v):
        return math.tan(v[self._children[0]])

    def d_v(self, index j, np.ndarray v):
        s = 1.0 / v[self._children[0]].cos()
        return s*s

    cdef float_t _d_v(self, index j, float_t[:] v):
        cdef float_t s = 1.0/math.cos(v[self._children[0]])
        return s*s

    def dd_vv(self, index j, index k, np.ndarray v):
        x = v[self._children[0]]
        s = 1.0/x.cos()
        return 2.0*s*s*x.tan()

    cdef float_t _dd_vv(self, index j, index k, float_t[:] v):
        cdef float_t x = v[self._children[0]]
        cdef float_t s = 1.0/math.cos(x)
        return 2.0*s*s*math.tan(x)


cdef class AsinExpression(UnaryFunctionExpression):
    def __init__(self, object children, object problem=None):
        super().__init__(children, UnaryFunctionType.Asin, problem)

    def eval(self, np.ndarray v):
        return v[self._children[0]].asin()

    cdef float_t _eval(self, float_t[:] v):
        return math.asin(v[self._children[0]])

    def d_v(self, index j, np.ndarray v):
        x = v[self._children[0]]
        return 1.0 / (1-x*x).sqrt()

    cdef float_t _d_v(self, index j, float_t[:] v):
        cdef float_t x = v[self._children[0]]
        return 1.0 / math.sqrt(1-x*x)

    def dd_vv(self, index j, index k, np.ndarray v):
        x = v[self._children[0]]
        return x / ((1-x*x) ** 1.5)

    cdef float_t _dd_vv(self, index j, index k, float_t[:] v):
        cdef float_t x = v[self._children[0]]
        return x / math.pow(1-x*x, 1.5)


cdef class AcosExpression(UnaryFunctionExpression):
    def __init__(self, object children, object problem=None):
        super().__init__(children, UnaryFunctionType.Acos, problem)

    def eval(self, np.ndarray v):
        return v[self._children[0]].acos()

    cdef float_t _eval(self, float_t[:] v):
        return math.acos(v[self._children[0]])

    def d_v(self, index j, np.ndarray v):
        x = v[self._children[0]]
        return -1.0 / (1-x*x).sqrt()

    cdef float_t _d_v(self, index j, float_t[:] v):
        cdef float_t x = v[self._children[0]]
        return -1.0 / math.sqrt(1-x*x)

    def dd_vv(self, index j, index k, np.ndarray v):
        x = v[self._children[0]]
        return -x / ((1-x*x) ** 1.5)

    cdef float_t _dd_vv(self, index j, index k, float_t[:] v):
        cdef float_t x = v[self._children[0]]
        return -x / math.pow(1-x*x, 1.5)


cdef class AtanExpression(UnaryFunctionExpression):
    def __init__(self, object children, object problem=None):
        super().__init__(children, UnaryFunctionType.Atan, problem)

    def eval(self, np.ndarray v):
        return v[self._children[0]].atan()

    cdef float_t _eval(self, float_t[:] v):
        return math.atan(v[self._children[0]])

    def d_v(self, index j, np.ndarray v):
        x = v[self._children[0]]
        return 1.0/(1+x*x)

    cdef float_t _d_v(self, index j, float_t[:] v):
        cdef float_t x = v[self._children[0]]
        return 1.0/(1+x*x)

    def dd_vv(self, index j, index k, np.ndarray v):
        x = v[self._children[0]]
        y = x*x
        return -(2*x) / ((1+y)*(1+y))

    cdef float_t _dd_vv(self, index j, index k, float_t[:] v):
        cdef float_t x = v[self._children[0]]
        cdef float_t y = x*x
        return -(2*x) / ((1+y)*(1+y))


cdef class Objective:
    def __init__(self, Expression root_expr, Sense sense):
        self.root_expr = root_expr
        self.sense = sense

    cpdef bint is_minimizing(self):
        return self.sense == Sense.MINIMIZE

    cpdef bint is_maximizing(self):
        return self.sense == Sense.MAXIMIZE


cdef class Constraint:
    def __init__(self, Expression root_expr, object lower_bound, object upper_bound):
        self.root_expr = root_expr
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound

    cpdef bint is_equality(self):
        return self.lower_bound == self.upper_bound


cdef class Variable(Expression):
    def __init__(self, object problem=None):
        super().__init__(ExpressionType.Variable, problem)
        self.default_depth = 0

    cpdef bint is_constant(self):
        if self.problem is not None:
            return self.problem.is_fixed(self)
        return 0

    cdef float_t _eval(self, float_t[:] v):
        return v[self.idx]


cdef class Constant(Expression):
    def __init__(self, float_t value, object problem=None):
        super().__init__(ExpressionType.Constant, problem)
        self.value = value
        # always come after variables
        self.default_depth = 1

    cpdef bint is_constant(self):
        return 1

    def eval(self, np.ndarray v):
        return self.value

    cdef float_t _eval(self, float_t[:] v):
        return self.value
