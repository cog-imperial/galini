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
    cdef index num_children
    cdef readonly index idx
    cdef index default_depth

    cdef reindex(self, index cutoff)
    cdef float_t _eval(self, float_t[:] v)
    cdef float_t _d_v(self, index j, float_t[:] v)
    cdef float_t _dd_vv(self, index j, index k, float_t[:] v)
    cpdef index nth_children(self, index i)


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
    cdef float_t *coefficients
    cdef float_t constant


cdef class UnaryFunctionExpression(UnaryExpression):
    cdef readonly str funct_name


cdef class NegationExpression(UnaryFunctionExpression):
    pass


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
    cdef Sense sense
    cdef Expression root_expr

    cpdef bint is_minimizing(self)
    cpdef bint is_maximizing(self)


cdef class Constraint:
    cdef object lower_bound
    cdef object upper_bound
    cdef Expression root_expr

    cpdef bint is_equality(self)


cdef class Variable(Expression):
    cdef Domain domain
    cdef object lower_bound
    cdef object upper_bound

    cpdef bint is_binary(self)
    cpdef bint is_integer(self)
    cpdef bint is_real(self)



cdef class Constant(Expression):
    cdef float_t value


cdef class Problem:
    cdef object vertices
    cdef readonly index size
    cdef index *depth
    cdef index depth_size

    cdef index num_variables
    cdef index num_constraints
    cdef index num_objectives

    cdef object _constraints
    cdef object _objectives
    cdef object _variables_by_name
    cdef object _constraints_by_name
    cdef object _objectives_by_name

    cpdef index max_depth(self)
    cpdef index vertex_depth(self, index i)

    cpdef Variable add_variable(self, str name, object lower_bound, object upper_bound, Domain domain)
    cpdef Variable variable(self, str name)

    cpdef Constraint add_constraint(self, str name, Expression root_expr, object lower_bound, object upper_bound)
    cpdef Constraint constraint(self, str name)

    cpdef Objective add_objective(self, str name, Expression root_expr, Sense sense)
    cpdef Objective objective(self, str name)

    cpdef insert_vertex(self, Expression expr)
    cdef _insert_vertex(self, Expression expr)
    cdef _realloc_depth(self)
