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
from cpython.mem cimport PyMem_Malloc, PyMem_Realloc, PyMem_Free

cdef index _bisect_left(index[:] depths, index n, index target):
    """Return insertion index for target depth.

    Returns index i such that

        max(depth[j] for j in j < i) <= depth

    and

        min(depth[j] for j in j > i) > depth
    """
    cdef index i
    if n == 0:
        return 0
    for i in range(n):
        if depths[i] > target:
            return i
    return n


cdef class VariableView:
    def __init__(self, Problem problem, Variable variable):
        self.problem = problem
        self.variable = variable

    cpdef void set_starting_point(self, float_t point):
        self.problem.set_starting_point(self.variable, point)

    cpdef bint has_starting_point(self):
        return self.problem.has_starting_point(self.variable)

    cpdef float_t starting_point(self):
        return self.problem.starting_point(self.variable)

    cpdef void set_value(self, float_t value):
        self.problem.set_value(self.variable, value)

    cpdef void unset_value(self):
        self.problem.unset_value(self.variable)

    cpdef bint has_value(self):
        return self.problem.has_value(self.variable)

    cpdef void fix(self, float_t point):
        self.problem.fix_variable(self.variable, point)

    cpdef void unfix(self):
        self.problem.unfix_variable(self.variable)

    cpdef bint is_fixed(self):
        return self.problem.is_fixed(self.variable)

    cpdef object lower_bound(self):
        return self.problem.variable_lower_bound(self.variable)

    cpdef void set_lower_bound(self, object value):
        self.problem.set_variable_lower_bound(self.variable, value)

    cpdef object upper_bound(self):
        return self.problem.variable_upper_bound(self.variable)

    cpdef void set_upper_bound(self, object value):
        self.problem.set_variable_upper_bound(self.variable, value)

    cpdef Domain _domain(self):
        return self.problem.variable_domain(self.variable)

    cpdef bint is_binary(self):
        return self._domain() == Domain.BINARY

    cpdef bint is_integer(self):
        return self._domain() == Domain.INTEGERS

    cpdef bint is_real(self):
        return self._domain() == Domain.REALS


cdef class Problem:
    # We have to explicitly implement all methods in the base class
    # RootProblem and ChildProblem will override them.
    def __init__(self):
        self.domains = []
        self.lower_bounds = []
        self.upper_bounds = []
        self.starting_points = []
        self.starting_points_mask = []
        self.values = []
        self.values_mask = []
        self.fixed = []
        self.fixed_mask = []

    cpdef index max_depth(self):
        pass

    cpdef index vertex_depth(self, index i):
        pass

    cpdef VariableView variable(self, str name):
        pass

    cpdef VariableView variable_at_index(self, index i):
        pass

    def variables(self):
        pass

    cpdef Constraint constraint(self, str name):
        pass

    cpdef Objective objective(self, str name):
        pass

    cpdef void set_starting_point(self, Variable v, float_t point):
        self.starting_points[v.idx] = point
        self.starting_points_mask[v.idx] = 1

    cpdef bint has_starting_point(self, Variable v):
        return self.starting_points_mask[v.idx]

    cpdef float_t starting_point(self, Variable v):
        return self.starting_points[v.idx]

    cpdef void set_value(self, Variable v, float_t value):
        self.values[v.idx] = value
        self.values_mask[v.idx] = 1

    cpdef void unset_value(self, Variable v):
        self.values_mask[v.idx] = 0

    cpdef bint has_value(self, Variable v):
        return self.values_mask[v.idx]

    cpdef void fix_variable(self, Variable v, float_t point):
        self.fixed[v.idx] = point
        self.fixed_mask[v.idx] = 1

    cpdef void unfix_variable(self, Variable v):
        self.fixed_mask[v.idx] = 0

    cpdef bint is_fixed(self, Variable v):
        return self.fixed_mask[v.idx]

    cpdef object variable_lower_bound(self, Variable v):
        return self.lower_bounds[v.idx]

    cpdef void set_variable_lower_bound(self, Variable v, object value):
        self.lower_bounds[v.idx] = value

    cpdef object variable_upper_bound(self, Variable v):
        return self.upper_bounds[v.idx]

    cpdef void set_variable_upper_bound(self, Variable v, object value):
        self.upper_bounds[v.idx] = value

    cpdef Domain variable_domain(self, Variable v):
        return self.domains[v.idx]

    cpdef ChildProblem make_child(self):
        pass

    @property
    def vertices(self):
        return None


cdef class RootProblem(Problem):
    def __init__(self, str name):
        super().__init__()
        self.name = name
        self._vertices = []
        self.size = 0

        starting_nodes = 512
        self.depth = <index *>PyMem_Malloc(starting_nodes * sizeof(index))
        if not self.depth:
            raise MemoryError()
        self.depth_size = starting_nodes

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
        cdef Variable var = Variable()
        self._insert_vertex(var)
        self._variables_by_name[name] = var
        self.num_variables += 1
        assert self.num_variables == len(self._variables_by_name)
        # new variables are always added after existing variables, so it is
        # safe to append their bounds and domains
        self.domains.append(domain)
        self.lower_bounds.append(lower_bound)
        self.upper_bounds.append(upper_bound)
        self.starting_points.append(0.0)
        self.starting_points_mask.append(0)
        self.values.append(0.0)
        self.values_mask.append(0)
        self.fixed.append(0.0)
        self.fixed_mask.append(0)
        return var

    cpdef VariableView variable(self, str name):
        cdef Variable var = self._variables_by_name[name]
        return VariableView(self, var)

    cpdef VariableView variable_at_index(self, index i):
        cdef Variable var = self._vertices[i]
        return VariableView(self, var)

    def variables(self):
        return self._variables_by_name

    def variables_view(self):
        for var in self.variables().values():
            yield VariableView(self, var)

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

    @property
    def constraints(self):
        return self._constraints_by_name

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

    @property
    def objectives(self):
        return self._objectives_by_name

    def first_child(self, Expression expr):
        cdef index idx = expr._nth_children(0)
        return self._vertices[idx]

    def second_child(self, Expression expr):
        cdef index idx = expr._nth_children(1)
        return self._vertices[idx]

    def nth_child(self, Expression expr, index i):
        cdef index idx = expr._nth_children(i)
        return self._vertices[idx]

    def children(self, Expression expr):
        cdef index i, idx
        cdef index n = expr.num_children
        for i in range(n):
            idx = expr._nth_children(i)
            yield self._vertices[idx]

    @property
    def vertices(self):
        return self._vertices

    def sorted_vertices(self):
        cdef index i
        for i in range(self.size):
            yield self._vertices[i]

    cpdef ChildProblem make_child(self):
        return ChildProblem(self)

    cpdef insert_vertex(self, Expression expr):
        if isinstance(expr, (Variable, Constraint, Objective)):
            raise RuntimeError('insert variable, constraint, objective')

        return self._insert_vertex(expr)

    cdef index _insert_vertex(self, Expression expr):
        cdef index i, children_idx, ins_idx
        cdef index depth
        cdef Expression cur_expr
        cdef index[:] depth_arr

        # special case for first element
        if self.size == 0:
            self._vertices.append(expr)
            expr.idx = 0
            self.depth[0] = expr.default_depth
            self.size = 1
            return 0

        if self.size >= self.depth_size:
            self._realloc_depth()

        depth_arr = <index[:self.depth_size]>self.depth
        # compute new expr depth
        depth = expr.default_depth
        for i in range(expr.num_children):
            children_idx = expr._nth_children(i)
            assert children_idx < self.size
            depth = max(depth, depth_arr[children_idx] + 1)

        # insert new node in vertices
        ins_idx = _bisect_left(depth_arr, self.size, depth)
        assert ins_idx <= self.size

        self._vertices.insert(ins_idx, expr)
        self.size += 1

        # shift depths right by 1
        for i in range(self.size, ins_idx - 1, -1):
            depth_arr[i+1] = depth_arr[i]
        depth_arr[ins_idx] = depth

        for i in range(self.size):
            cur_expr = self._vertices[i]
            cur_expr.reindex(ins_idx)
        expr.idx = ins_idx
        return ins_idx

    cdef void _realloc_depth(self):
        cdef index new_size = 2 * self.depth_size
        cdef index *new_depth = <index *>PyMem_Realloc(self.depth, new_size * sizeof(index))
        if not new_depth:
            raise MemoryError()
        self.depth = new_depth
        self.depth_size = new_size


cdef class ChildProblem(Problem):
    def __init__(self, Problem parent):
        super().__init__()
        self.parent = parent

        self.num_variables = parent.num_variables
        self.num_constraints = parent.num_constraints
        self.num_objectives = parent.num_objectives

        self._init_variables_storage()

    def _init_variables_storage(self):
        self.domains = self.parent.domains.copy()
        self.lower_bounds = self.parent.lower_bounds.copy()
        self.upper_bounds = self.parent.upper_bounds.copy()
        self.starting_points = self.parent.starting_points.copy()
        self.starting_points_mask = self.parent.starting_points_mask.copy()
        self.values = self.parent.values.copy()
        self.values_mask = self.parent.values_mask.copy()
        self.fixed = self.parent.fixed.copy()
        self.fixed_mask = self.parent.fixed_mask.copy()


    cpdef VariableView variable(self, str name):
        cdef Variable var = self.parent._variables_by_name[name]
        return VariableView(self, var)

    cpdef VariableView variable_at_index(self, index i):
        cdef Variable var = self.parent.vertices[i]
        return VariableView(self, var)

    def variables(self):
        return self.parent.variables()

    def variables_view(self):
        for var in self.variables().values():
            yield VariableView(self, var)

    cpdef ChildProblem make_child(self):
        return ChildProblem(self)

    @property
    def vertices(self):
        return self.parent.vertices
