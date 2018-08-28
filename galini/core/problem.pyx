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


cdef class Problem:
    def __init__(self, str name):
        self.name = name
        self.vertices = []
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
        cdef Variable var = Variable(lower_bound, upper_bound, domain)
        self._insert_vertex(var)
        self._variables_by_name[name] = var
        self.num_variables += 1
        assert self.num_variables == len(self._variables_by_name)
        return var

    cpdef Variable variable(self, str name):
        return self._variables_by_name[name]

    @property
    def variables(self):
        return self._variables_by_name

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
        return self.vertices[idx]

    def second_child(self, Expression expr):
        cdef index idx = expr._nth_children(1)
        return self.vertices[idx]

    def nth_child(self, Expression expr, index i):
        cdef index idx = expr._nth_children(i)
        return self.vertices[idx]

    def children(self, Expression expr):
        cdef index i, idx
        cdef index n = expr.num_children
        for i in range(n):
            idx = expr._nth_children(i)
            yield self.vertices[idx]

    def sorted_vertices(self):
        cdef index i
        for i in range(self.size):
            yield self.vertices[i]

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
            self.vertices.append(expr)
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

        self.vertices.insert(ins_idx, expr)
        self.size += 1

        # shift depths right by 1
        for i in range(self.size, ins_idx - 1, -1):
            depth_arr[i+1] = depth_arr[i]
        depth_arr[ins_idx] = depth

        for i in range(self.size):
            cur_expr = self.vertices[i]
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
