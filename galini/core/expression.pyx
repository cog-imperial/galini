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

cdef class Expression:
    def __init__(self, int expression_type):
        self.num_children = 0
        self.idx = 0
        # always come after variables (0) and constants(1)
        self.default_depth = 2
        self.expression_type = expression_type

    cdef void reindex(self, index cutoff) nogil:
        if self.idx >= cutoff:
            self.idx += 1

    cpdef foo eval(self, foo[:] v):
        return self._eval(v)

    cdef foo _eval(self, foo[:] v):
        return 0.0

    cpdef foo d_v(self, index j, foo[:] v):
        return self._d_v(j, v)

    cdef foo _d_v(self, index j, foo[:] v):
        return 0.0

    cpdef foo dd_vv(self, index j, index k, foo[:] v):
        return self._dd_vv(j, k, v)

    cdef foo _dd_vv(self, index j, index k, foo[:] v):
        return 0.0

    cpdef index nth_children(self, index i):
        return self._nth_children(i)

    cdef index _nth_children(self, index i) nogil:
        return 0


cdef class UnaryExpression(Expression):
    def __init__(self, object children, int expression_type):
        super().__init__(expression_type)
        assert len(children) == 1
        self.num_children = 1
        self.children[0] = children[0]

    cdef void reindex(self, index cutoff) nogil:
        cdef index children_idx
        if self.idx >= cutoff:
            self.idx += 1

        children_idx = self.children[0]
        if children_idx >= cutoff:
            self.children[0] = children_idx + 1

    cdef index _nth_children(self, index i) nogil:
        return self.children[0]


cdef class BinaryExpression(Expression):
    def __init__(self, object children, int expression_type):
        super().__init__(expression_type)
        assert len(children) == 2
        self.num_children = 2
        self.children[0] = children[0]
        self.children[1] = children[1]

    cdef void reindex(self, index cutoff) nogil:
        cdef index children_idx
        if self.idx >= cutoff:
            self.idx += 1

        for i in range(2):
            children_idx = self.children[i]
            if children_idx >= cutoff:
                self.children[i] = children_idx + 1

    cdef index _nth_children(self, index i) nogil:
        return self.children[i]


cdef class NaryExpression(Expression):
    def __init__(self, object children, int expression_type):
        super().__init__(expression_type)

        cdef index num_children = len(children)
        assert num_children > 0
        self.children = <index *>PyMem_Malloc(num_children * sizeof(index))
        self.num_children = num_children

        cdef index i
        for i in range(num_children):
            self.children[i] = children[i]

    def __dealloc__(self):
        PyMem_Free(self.children)

    cdef void reindex(self, index cutoff) nogil:
        cdef index children_idx
        if self.idx >= cutoff:
            self.idx += 1

        for i in range(self.num_children):
            children_idx = self.children[i]
            if children_idx >= cutoff:
                self.children[i] = children_idx + 1

    cdef index _nth_children(self, index i) nogil:
        return self.children[i]


cdef class ProductExpression(BinaryExpression):
    def __init__(self, object children):
        super().__init__(children, ExpressionType.Product)

    cdef foo _eval(self, foo[:] v):
        return v[self.children[0]] * v[self.children[1]]

    cdef foo _d_v(self, index j, foo[:] v):
        if j == 0:
            return v[self.children[1]]
        elif j == 1:
            return v[self.children[0]]
        return INFINITY


    cdef foo _dd_vv(self, index j, index k, foo[:] v):
        if j == k:
            return 0.0
        else:
            return 1.0


cdef class DivisionExpression(BinaryExpression):
    def __init__(self, object children):
        super().__init__(children, ExpressionType.Division)

    cdef foo _eval(self, foo[:] v):
        return v[self.children[0]] / v[self.children[1]]

    cdef foo _d_v(self, index j, foo[:] v):
        cdef float_t y
        if j == 0:
            # d/dx (x/y) := 1/y
            return 1.0/v[self.children[1]]
        elif j == 1:
            # d/dy (x/y) := -x/y^2
            y = v[self.children[1]]
            return -v[self.children[0]] / (y*y)
        else:
            return INFINITY

    cdef foo _dd_vv(self, index j, index k, foo[:] v):
        cdef float_t x
        cdef float_t y = v[self.children[1]]
        if j == k:
            if j == 0:
                # d/dx^2
                return 0.0
            else:
                x = v[self.children[0]]
                return 2*x / (y*y*y)
        else:
            # d/dxdy == d/dydx == -1/y^2
            return -1.0 / (y*y)



cdef class SumExpression(NaryExpression):
    def __init__(self, object children):
        super().__init__(children, ExpressionType.Sum)

    cdef foo _eval(self, foo[:] v):
        cdef index i
        cdef float_t tot = 0
        for i in range(self.num_children):
            tot += v[self.children[i]]
        return tot

    cdef foo _d_v(self, index j, foo[:] v):
        return 1.0

    cdef foo _dd_vv(self, index j, index k, foo[:] v):
        return 0.0


cdef class PowExpression(BinaryExpression):
    def __init__(self, object children):
        super().__init__(children, ExpressionType.Power)

    cdef foo _eval(self, foo[:] v):
        cdef float_t base = v[self.children[0]]
        cdef float_t expo = v[self.children[1]]
        if expo == 0:
            return 0.0
        elif expo == 1:
            return base
        elif expo == 2:
            return base*base
        else:
            return math.pow(base, expo)

    cdef foo _d_v(self, index j, foo[:] v):
        cdef float_t base = v[self.children[0]]
        cdef float_t expo = v[self.children[1]]
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

    cdef foo _dd_vv(self, index j, index k, foo[:] v):
        cdef float_t a, l, b, c
        cdef float_t base = v[self.children[0]]
        cdef float_t expo = v[self.children[1]]
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
    def __init__(self, object children, float_t[:] coefficients, float_t constant):
        super().__init__(children, ExpressionType.Linear)

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

    cdef foo _eval(self, foo[:] v):
        cdef index i
        cdef float_t tot = self.constant
        for i in range(self.num_children):
            tot += self._coefficients[i] * v[self.children[i]]
        return tot

    cdef foo _d_v(self, index j, foo[:] v):
        return self._coefficients[j]

    cdef foo _dd_vv(self, index j, index k, foo[:] v):
        return 0.0


cdef class NegationExpression(UnaryExpression):
    def __init__(self, object children):
        super().__init__(children, ExpressionType.Negation)

    cdef foo _eval(self, foo[:] v):
        return -v[self.children[0]]

    cdef foo _d_v(self, index j, foo[:] v):
        return -1.0


cdef class UnaryFunctionExpression(UnaryExpression):
    def __init__(self, object children, int func_type):
        super().__init__(children, ExpressionType.UnaryFunction)
        self.func_type = func_type


cdef class AbsExpression(UnaryFunctionExpression):
    def __init__(self, object children):
        super().__init__(children, UnaryFunctionType.Abs)

    cdef foo _eval(self, foo[:] v):
        cdef float_t x = v[self.children[0]]
        if x >= 0:
            return x
        else:
            return -x

    cdef foo _d_v(self, index j, foo[:] v):
        cdef float_t x = v[self.children[0]]
        if x >= 0:
            return 1.0
        else:
            return -1.0


cdef class SqrtExpression(UnaryFunctionExpression):
    def __init__(self, object children):
        super().__init__(children, UnaryFunctionType.Sqrt)

    cdef foo _eval(self, foo[:] v):
        return math.sqrt(v[self.children[0]])

    cdef foo _d_v(self, index j, foo[:] v):
        return 1.0 / (2.0 * math.sqrt(v[self.children[0]]))

    cdef foo _dd_vv(self, index j, index k, foo[:] v):
        cdef float_t s = math.sqrt(v[self.children[0]])
        return -1.0/(4.0 * math.pow(s, 3))


cdef class ExpExpression(UnaryFunctionExpression):
    def __init__(self, object children):
        super().__init__(children, UnaryFunctionType.Exp)

    cdef foo _eval(self, foo[:] v):
        return math.exp(v[self.children[0]])

    cdef foo _d_v(self, index j, foo[:] v):
        return math.exp(v[self.children[0]])

    cdef foo _dd_vv(self, index j, index k, foo[:] v):
        return math.exp(v[self.children[0]])


cdef class LogExpression(UnaryFunctionExpression):
    def __init__(self, object children):
        super().__init__(children, UnaryFunctionType.Log)

    cdef foo _eval(self, foo[:] v):
        return math.log(v[self.children[0]])

    cdef foo _d_v(self, index j, foo[:] v):
        return 1.0 / v[self.children[0]]

    cdef foo _dd_vv(self, index j, index k, foo[:] v):
        cdef float_t x = v[self.children[0]]
        return -1.0 / (x*x)


cdef class SinExpression(UnaryFunctionExpression):
    def __init__(self, object children):
        super().__init__(children, UnaryFunctionType.Sin)

    cdef foo _eval(self, foo[:] v):
        return math.sin(v[self.children[0]])

    cdef foo _d_v(self, index j, foo[:] v):
        return math.cos(v[self.children[0]])

    cdef foo _dd_vv(self, index j, index k, foo[:] v):
        return -math.sin(v[self.children[0]])


cdef class CosExpression(UnaryFunctionExpression):
    def __init__(self, object children):
        super().__init__(children, UnaryFunctionType.Cos)

    cdef foo _eval(self, foo[:] v):
        return math.cos(v[self.children[0]])

    cdef foo _d_v(self, index j, foo[:] v):
        return -math.sin(v[self.children[0]])

    cdef foo _dd_vv(self, index j, index k, foo[:] v):
        return -math.cos(v[self.children[0]])


cdef class TanExpression(UnaryFunctionExpression):
    def __init__(self, object children):
        super().__init__(children, UnaryFunctionType.Tan)

    cdef foo _eval(self, foo[:] v):
        return math.tan(v[self.children[0]])

    cdef foo _d_v(self, index j, foo[:] v):
        cdef float_t s = 1.0/math.cos(v[self.children[0]])
        return s*s

    cdef foo _dd_vv(self, index j, index k, foo[:] v):
        cdef float_t x = v[self.children[0]]
        cdef float_t s = 1.0/math.cos(x)
        return 2.0*s*s*math.tan(x)


cdef class AsinExpression(UnaryFunctionExpression):
    def __init__(self, object children):
        super().__init__(children, UnaryFunctionType.Asin)

    cdef foo _eval(self, foo[:] v):
        return math.asin(v[self.children[0]])

    cdef foo _d_v(self, index j, foo[:] v):
        cdef float_t x = v[self.children[0]]
        return 1.0 / math.sqrt(1-x*x)

    cdef foo _dd_vv(self, index j, index k, foo[:] v):
        cdef float_t x = v[self.children[0]]
        return x / math.pow(1-x*x, 1.5)


cdef class AcosExpression(UnaryFunctionExpression):
    def __init__(self, object children):
        super().__init__(children, UnaryFunctionType.Acos)

    cdef foo _eval(self, foo[:] v):
        return math.acos(v[self.children[0]])

    cdef foo _d_v(self, index j, foo[:] v):
        cdef float_t x = v[self.children[0]]
        return -1.0 / math.sqrt(1-x*x)

    cdef foo _dd_vv(self, index j, index k, foo[:] v):
        cdef float_t x = v[self.children[0]]
        return -x / math.pow(1-x*x, 1.5)


cdef class AtanExpression(UnaryFunctionExpression):
    def __init__(self, object children):
        super().__init__(children, UnaryFunctionType.Atan)

    cdef foo _eval(self, foo[:] v):
        return math.atan(v[self.children[0]])

    cdef foo _d_v(self, index j, foo[:] v):
        cdef float_t x = v[self.children[0]]
        return 1.0/(1+x*x)

    cdef foo _dd_vv(self, index j, index k, foo[:] v):
        cdef float_t x = v[self.children[0]]
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
    def __init__(self, object lower_bound, object upper_bound, Domain domain):
        super().__init__(ExpressionType.Variable)
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound
        self.domain = domain
        self.default_depth = 0
        self.has_starting_point = 0
        self.starting_point = 0.0
        self.value = 0
        self.has_value = 0
        self.is_fixed = 0

    cpdef void set_starting_point(self, float_t point):
        self.starting_point = point
        self.has_starting_point = 1

    cpdef void set_value(self, float_t value):
        self.value = value
        self.has_value = 1

    cpdef void unset_value(self):
        self.has_value = 0

    cpdef void fix(self, float_t point):
        self.value = point
        self.is_fixed = 1

    cpdef void unfix(self):
        self.is_fixed = 0

    cpdef bint is_binary(self):
        return self.domain == Domain.BINARY

    cpdef bint is_integer(self):
        return self.domain == Domain.INTEGERS

    cpdef bint is_real(self):
        return self.domain == Domain.REALS

    cdef foo _eval(self, foo[:] v):
        return v[self.idx]


cdef class Constant(Expression):
    def __init__(self, float_t value):
        super().__init__(ExpressionType.Constant)
        self.value = value
        # always come after variables
        self.default_depth = 1

    cdef foo _eval(self, foo[:] v):
        return self.value
