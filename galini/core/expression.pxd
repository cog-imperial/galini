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
cimport numpy as np

ctypedef np.uint32_t index
ctypedef np.float64_t float_t


cpdef enum Domain:
    REALS = 0
    INTEGERS = 1
    BINARY = 2


cpdef enum Sense:
    MINIMIZE = 0
    MAXIMIZE = 1


cdef class Expression:
    cdef readonly index num_children
    cdef readonly index idx
    cdef index default_depth
    cdef readonly int expression_type

    cdef void reindex(self, index cutoff) nogil
    cdef float_t _eval(self, float_t[:] v)
    cdef float_t _d_v(self, index j, float_t[:] v)
    cdef float_t _dd_vv(self, index j, index k, float_t[:] v)
    cpdef index nth_children(self, index i)
    cdef index _nth_children(self, index i) nogil


cdef class UnaryExpression(Expression):
    cdef index children[1]


cdef class BinaryExpression(Expression):
    cdef index children[2]


cdef class NaryExpression(Expression):
    cdef index *children


cdef class ProductExpression(BinaryExpression):
    pass


cdef class DivisionExpression(BinaryExpression):
    pass


cdef class SumExpression(NaryExpression):
    pass


cdef class PowExpression(BinaryExpression):
    pass


cdef class LinearExpression(NaryExpression):
    cdef float_t *_coefficients
    cdef readonly float_t constant


cdef class NegationExpression(UnaryExpression):
    pass


cdef class UnaryFunctionExpression(UnaryExpression):
    cdef readonly int func_type


cdef class AbsExpression(UnaryFunctionExpression):
    pass


cdef class SqrtExpression(UnaryFunctionExpression):
    pass


cdef class ExpExpression(UnaryFunctionExpression):
    pass


cdef class LogExpression(UnaryFunctionExpression):
    pass


cdef class SinExpression(UnaryFunctionExpression):
    pass


cdef class CosExpression(UnaryFunctionExpression):
    pass


cdef class TanExpression(UnaryFunctionExpression):
    pass


cdef class AsinExpression(UnaryFunctionExpression):
    pass


cdef class AcosExpression(UnaryFunctionExpression):
    pass


cdef class AtanExpression(UnaryFunctionExpression):
    pass


cdef class Objective:
    cdef readonly Sense sense
    cdef readonly Expression root_expr

    cpdef bint is_minimizing(self)
    cpdef bint is_maximizing(self)


cdef class Constraint:
    cdef readonly object lower_bound
    cdef readonly object upper_bound
    cdef readonly Expression root_expr

    cpdef bint is_equality(self)


cdef class Variable(Expression):
    pass


cdef class Constant(Expression):
    cdef readonly float_t value
