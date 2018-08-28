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
from galini.core.expression cimport (
    index,
    Sense,
    Domain,
    Expression,
    Variable,
    Constraint,
    Objective,
)


cdef class Problem:
    cdef readonly str name
    cdef object vertices
    cdef readonly index size
    cdef index *depth
    cdef index depth_size

    cdef readonly index num_variables
    cdef readonly index num_constraints
    cdef readonly index num_objectives

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
    cdef index _insert_vertex(self, Expression expr)
    cdef void _realloc_depth(self)
