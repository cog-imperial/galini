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
from galini.core.expression cimport (
    float_t,
    index,
    Sense,
    Domain,
    Expression,
    Variable,
    Constraint,
    Objective,
)


cdef class VariableView:
    cdef readonly Problem problem
    cdef readonly Variable variable

    cpdef bint is_binary(self)
    cpdef bint is_integer(self)
    cpdef bint is_real(self)
    cpdef void set_starting_point(self, float_t point)
    cpdef bint has_starting_point(self)
    cpdef float_t starting_point(self)
    cpdef void set_value(self, float_t value)
    cpdef void unset_value(self)
    cpdef bint has_value(self)
    cpdef void fix(self, float_t point)
    cpdef void unfix(self)
    cpdef bint is_fixed(self)
    cpdef object lower_bound(self)
    cpdef void set_lower_bound(self, object value)
    cpdef object upper_bound(self)
    cpdef void set_upper_bound(self, object value)

    cpdef Domain _domain(self)


cdef class Problem:
    cdef readonly index num_variables
    cdef readonly index num_constraints
    cdef readonly index num_objectives

    # Use Python lists since they are not used frequently enough to be
    # worth managing the memory ourselves

    # Variables domain
    cdef readonly object domains
    # Variables lower and upper bounds
    cdef readonly object lower_bounds
    cdef readonly object upper_bounds
    # Variables starting points and mask
    cdef readonly object starting_points
    cdef readonly object starting_points_mask
    # Variables values from solution and mask
    cdef readonly object values
    cdef readonly object values_mask
    # Variable mask for fixed variables
    cdef readonly object fixed
    cdef readonly object fixed_mask

    cpdef index max_depth(self)
    cpdef index vertex_depth(self, index i)

    cpdef VariableView variable(self, str name)
    cpdef VariableView variable_at_index(self, index i)
    cpdef Constraint constraint(self, str name)
    cpdef Objective objective(self, str name)

    cpdef void set_starting_point(self, Variable v, float_t point)
    cpdef bint has_starting_point(self, Variable v)
    cpdef float_t starting_point(self, Variable v)
    cpdef void set_value(self, Variable v, float_t value)
    cpdef void unset_value(self, Variable v)
    cpdef bint has_value(self, Variable v)
    cpdef void fix_variable(self, Variable v, float_t point)
    cpdef void unfix_variable(self, Variable v)
    cpdef bint is_fixed(self, Variable v)
    cpdef object variable_lower_bound(self, Variable v)
    cpdef void set_variable_lower_bound(self, Variable v, object value)
    cpdef object variable_upper_bound(self, Variable v)
    cpdef void set_variable_upper_bound(self, Variable v, object value)
    cpdef Domain variable_domain(self, Variable v)

    cpdef ChildProblem make_child(self)


cdef class RootProblem(Problem):
    cdef readonly str name
    cdef readonly object _vertices
    cdef readonly index size
    cdef index *depth
    cdef index depth_size

    cdef readonly object _constraints
    cdef readonly object _objectives
    cdef object _variables_by_name
    cdef object _constraints_by_name
    cdef object _objectives_by_name

    cpdef Variable add_variable(self, str name, object lower_bound, object upper_bound, Domain domain)
    cpdef Constraint add_constraint(self, str name, Expression root_expr, object lower_bound, object upper_bound)
    cpdef Objective add_objective(self, str name, Expression root_expr, Sense sense)

    cpdef insert_vertex(self, Expression expr)
    cdef index _insert_vertex(self, Expression expr)
    cdef void _realloc_depth(self)


cdef class ChildProblem(Problem):
    cdef readonly Problem parent
