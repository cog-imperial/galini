# cython: cdivision=True, boundscheck=False, wraparound=False
from cpython.mem cimport PyMem_Malloc, PyMem_Realloc, PyMem_Free
cimport libc.math as math
cimport numpy as np
import numpy as np


float_ = np.float64
index_ = np.uint32


cdef class Expression:
    def __init__(self):
        self.num_children = 0
        self.idx = 0
        # always come after variables (0) and constants(1)
        self.default_depth = 2

    cdef void reindex(self, index cutoff) nogil:
        if self.idx >= cutoff:
            self.idx += 1

    cdef float_t _eval(self, float_t[:] v) nogil:
        return 0.0

    cdef float_t _d_v(self, index j, float_t[:] v) nogil:
        return 0.0

    cdef float_t _dd_vv(self, index j, index k, float_t[:] v) nogil:
        return 0.0

    cpdef index nth_children(self, index i):
        return self._nth_children(i)

    cdef index _nth_children(self, index i) nogil:
        return 0


cdef class UnaryExpression(Expression):
    def __init__(self, object children):
        super().__init__()
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
    def __init__(self, object children):
        super().__init__()
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
    def __init__(self, object children):
        super().__init__()

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
    cdef float_t _eval(self, float_t[:] v) nogil:
        return v[self.children[0]] * v[self.children[1]]

    cdef float_t _d_v(self, index j, float_t[:] v) nogil:
        if j == self.children[0]:
            return v[self.children[1]]
        else:
            return v[self.children[0]]

    cdef float_t _dd_vv(self, index j, index k, float_t[:] v) nogil:
        if j == k:
            return 0.0
        else:
            return 1.0


cdef class DivisionExpression(BinaryExpression):
    cdef float_t _eval(self, float_t[:] v) nogil:
        return v[self.children[0]] / v[self.children[1]]

    cdef float_t _d_v(self, index j, float_t[:] v) nogil:
        cdef float_t y
        if j == self.children[0]:
            # d/dx (x/y) := 1/y
            return 1.0/v[self.children[1]]
        else:
            # d/dy (x/y) := -x/y^2
            y = self.children[1]
            return -self.children[0] / (y*y)

    cdef float_t _dd_vv(self, index j, index k, float_t[:] v) nogil:
        return 0.0



cdef class SumExpression(NaryExpression):
    cdef float_t _eval(self, float_t[:] v) nogil:
        cdef index i
        cdef float_t tot = 0
        for i in range(self.num_children):
            tot += v[self.children[i]]
        return tot

    cdef float_t _d_v(self, index j, float_t[:] v) nogil:
        return 1.0

    cdef float_t _dd_vv(self, index j, index k, float_t[:] v) nogil:
        return 0.0


cdef class PowExpression(BinaryExpression):
    cdef float_t _eval(self, float_t[:] v) nogil:
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


cdef class LinearExpression(NaryExpression):
    def __init__(self, object children, float_t[:] coefficients, float_t constant):
        super().__init__(children)

        self.constant = constant
        self.coefficients = <float_t *>PyMem_Malloc(self.num_children * sizeof(index))
        cdef float_t[:] coefficients_view = <float_t[:self.num_children]>self.coefficients
        coefficients_view[:] = coefficients

    cdef float_t _eval(self, float_t[:] v) nogil:
        cdef index i
        cdef float_t tot = self.constant
        for i in range(self.num_children):
            tot += self.coefficients[i] * v[self.children[i]]
        return tot

    cdef float_t _d_v(self, index j, float_t[:] v) nogil:
        return self.coefficients[j]

    cdef float_t _dd_vv(self, index j, index k, float_t[:] v) nogil:
        return 0.0


cdef class UnaryFunctionExpression(UnaryExpression):
    def __init__(self, object children, str funct_name):
        super().__init__(children)
        self.funct_name = funct_name


cdef class NegationExpression(UnaryFunctionExpression):
    def __init__(self, object children, str funct_name):
        super().__init__(children, 'negation')

    cdef float_t _eval(self, float_t[:] v) nogil:
        return -v[self.children[0]]


cdef class AbsExpression(UnaryFunctionExpression):
    def __init__(self, object children, str funct_name):
        super().__init__(children, 'abs')

    cdef float_t _eval(self, float_t[:] v) nogil:
        cdef float_t x = v[self.children[0]]
        if x >= 0:
            return x
        else:
            return -x


cdef class SqrtExpression(UnaryFunctionExpression):
    def __init__(self, object children, str funct_name):
        super().__init__(children, 'sqrt')

    cdef float_t _eval(self, float_t[:] v) nogil:
        return math.sqrt(v[self.children[0]])


cdef class ExpExpression(UnaryFunctionExpression):
    def __init__(self, object children, str funct_name):
        super().__init__(children, 'exp')

    cdef float_t _eval(self, float_t[:] v) nogil:
        return math.exp(v[self.children[0]])


cdef class LogExpression(UnaryFunctionExpression):
    def __init__(self, object children, str funct_name):
        super().__init__(children, 'log')

    cdef float_t _eval(self, float_t[:] v) nogil:
        return math.log(v[self.children[0]])


cdef class SinExpression(UnaryFunctionExpression):
    def __init__(self, object children, str funct_name):
        super().__init__(children, 'sin')

    cdef float_t _eval(self, float_t[:] v) nogil:
        return math.sin(v[self.children[0]])


cdef class CosExpression(UnaryFunctionExpression):
    def __init__(self, object children, str funct_name):
        super().__init__(children, 'cos')

    cdef float_t _eval(self, float_t[:] v) nogil:
        return math.cos(v[self.children[0]])


cdef class TanExpression(UnaryFunctionExpression):
    def __init__(self, object children, str funct_name):
        super().__init__(children, 'tan')

    cdef float_t _eval(self, float_t[:] v) nogil:
        return math.tan(v[self.children[0]])


cdef class AsinExpression(UnaryFunctionExpression):
    def __init__(self, object children, str funct_name):
        super().__init__(children, 'asin')

    cdef float_t _eval(self, float_t[:] v) nogil:
        return math.asin(v[self.children[0]])


cdef class AcosExpression(UnaryFunctionExpression):
    def __init__(self, object children, str funct_name):
        super().__init__(children, 'acos')

    cdef float_t _eval(self, float_t[:] v) nogil:
        return math.acos(v[self.children[0]])


cdef class AtanExpression(UnaryFunctionExpression):
    def __init__(self, object children, str funct_name):
        super().__init__(children, 'atan')

    cdef float_t _eval(self, float_t[:] v) nogil:
        return math.atan(v[self.children[0]])


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
        super().__init__()
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound
        self.domain = domain
        self.default_depth = 0

    cpdef bint is_binary(self):
        return self.domain == Domain.BINARY

    cpdef bint is_integer(self):
        return self.domain == Domain.INTEGERS

    cpdef bint is_real(self):
        return self.domain == Domain.REALS

    cdef float_t _eval(self, float_t[:] v) nogil:
        return v[self.idx]


cdef class Constant(Expression):
    def __init__(self, float_t value):
        super().__init__()
        self.value = value
        # always come after variables
        self.default_depth = 1

    cdef float_t _eval(self, float_t[:] v) nogil:
        return self.value


cdef index _bisect_left(index[:] depths, index target):
    """Return insertion index for target depth.

    Returns index i such that

        max(depth[j] for j in j < i) <= depth

    and

        min(depth[j] for j in j > i) > depth
    """
    cdef index i
    cdef index n = len(depths)
    for i in range(n):
        if depths[i] > target:
            return i
    return n - 1


cdef class Problem:
    def __init__(self):
        self.vertices = []
        self.size = 0

        STARTING_NODES = 512
        self.depth = <index *>PyMem_Malloc(STARTING_NODES * sizeof(index))
        if not self.depth:
            raise MemoryError()
        self.depth_size = STARTING_NODES

        self.num_variables = 0
        self.num_constraints = 0
        self.num_objectives = 0

        self._constraints = []
        self._objectives = []
        self._variables_by_name = {}
        self._constraints_by_name = {}
        self._objectives_by_name = {}

    def __dealloc__(self):
        PyMem_Free(self.depth)

    cpdef index max_depth(self):
        if self.size == 0:
            return 0
        else:
            return self.depth[self.size-1]

    cpdef index vertex_depth(self, index i):
        assert i < self.size
        return self.depth[i]

    cpdef Variable add_variable(self, str name, object lower_bound, object upper_bound, Domain domain):
        if name in self._variables_by_name:
            raise RuntimeError('variable {} already exists'.format(name))
        cdef Variable var = Variable(lower_bound, upper_bound, domain)
        self._insert_vertex(var)
        self._variables_by_name[name] = var
        self.num_variables += 1
        assert self.num_variables == len(self._variables_by_name)
        return var

    cpdef Variable variable(self, str name):
        return self._variables_by_name[name]

    cpdef Constraint add_constraint(self, str name, Expression root_expr, object lower_bound, object upper_bound):
        if name in self._constraints_by_name:
            raise RuntimeError('constraint {} already exists'.format(name))
        cdef Constraint cons = Constraint(root_expr, lower_bound, upper_bound)
        self._constraints.append(cons)
        self._constraints_by_name[name] = cons
        self.num_constraints += 1
        assert self.num_constraints == len(self._constraints) == len(self._constraints_by_name)
        return cons

    cpdef Constraint constraint(self, str name):
        return self._constraints_by_name[name]

    cpdef Objective add_objective(self, str name, Expression root_expr, Sense sense):
        if name in self._objectives_by_name:
            raise RuntimeError('objective {} already exists'.format(name))
        cdef Objective obj = Objective(root_expr, sense)
        self._objectives.append(obj)
        self._objectives_by_name[name] = obj
        self.num_objectives += 1
        assert self.num_objectives == len(self._objectives) == len(self._objectives_by_name)
        return obj

    cpdef Objective objective(self, str name):
        return self._objectives_by_name[name]

    cpdef insert_vertex(self, Expression expr):
        if isinstance(expr, (Variable, Constraint, Objective)):
            raise RuntimeError('insert variable, constraint, objective')

        self._insert_vertex(expr)

    cdef _insert_vertex(self, Expression expr):
        cdef index i, children_idx, ins_idx
        cdef Expression cur_expr
        if self.size >= self.depth_size:
            self._realloc_depth()

        # compute new expr depth
        depth = expr.default_depth
        for i in range(expr.num_children):
            children_idx = expr._nth_children(i)
            assert children_idx < self.size
            depth = max(depth, self.depth[children_idx] + 1)

        # insert new node in vertices
        ins_idx = _bisect_left(<index[:self.depth_size]>self.depth, depth)

        self.vertices.insert(ins_idx, expr)
        self.size += 1
        # shift depths right by 1
        for i in range(self.depth_size, ins_idx - 1, -1):
            self.depth[i+1] = self.depth[i]
        self.depth[ins_idx] = depth

        for i in range(self.size):
            cur_expr = self.vertices[i]
            cur_expr.reindex(ins_idx)
        expr.idx = ins_idx

    cdef _realloc_depth(self):
        cdef index new_size = 2 * self.depth_size
        cdef index *new_depth = <index *>PyMem_Realloc(self.depth, new_size * sizeof(index))
        if not new_depth:
            raise MemoryError()
        self.depth = new_depth
