#  Copyright 2019 Francesco Ceccon
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.

"""Branch & Cut node storage. Contains original relaxed problem."""
import pyomo.environ as pe
from coramin.relaxations import relaxation_data_objects
from coramin.relaxations.univariate import PWXSquaredRelaxationData
from pyomo.contrib.fbbt.fbbt import compute_bounds_on_expr

from galini.branch_and_bound.branching import branch_at_point
from galini.cuts.pool import CutNodeStorage, CutPool
from galini.pyomo import safe_setlb, safe_setub
from galini.relaxations.relax import RelaxationData


class BranchingDecision:
    __slots__ = ('variable', 'point')

    def __init__(self, variable, point):
        self.variable = variable
        self.point = point

    def __str__(self):
        return 'BranchingDecision(variable={}, point={})'.format(
            self.variable, self.point
        )


class _NodeStorageBase:
    def __init__(self, root, parent, bounds):
        self.root = root
        self.parent = parent
        self._bounds = bounds
        self.branching_decision = None
        self.branching_point = None
        self.cut_pool = None
        self.cut_node_storage = None

    @property
    def is_root(self):
        return None

    @property
    def model_bounds(self):
        return self._bounds

    def branching_data(self):
        return self.model()

    def update_bounds(self, bounds):
        for var in self._bounds.keys():
            var_bounds = bounds.get(var, None)
            if var_bounds is None:
                continue
            self._bounds[var] = (var_bounds.lower_bound, var_bounds.upper_bound)

    def model(self):
        for var, (lb, ub) in self._bounds.items():
            safe_setlb(var, lb)
            safe_setub(var, ub)
        return self.root._model

    def model_relaxation(self):
        self.recompute_model_relaxation_bounds()
        return self.root._linear_model

    @property
    def relaxation_data(self):
        return self.root._relaxation_data

    def recompute_model_relaxation_bounds(self):
        linear_model = self.root._linear_model
        for var, (lb, ub) in self._bounds.items():
            linear_var = self.root.model_to_relaxation_var_map[var]
            safe_setlb(linear_var, lb)
            safe_setub(linear_var, ub)

        for relaxation in relaxation_data_objects(linear_model, active=True, descend_into=True):
            aux_var = relaxation.get_aux_var()
            rhs_expr = relaxation.get_rhs_expr()
            new_lb, new_ub = compute_bounds_on_expr(rhs_expr)
            safe_setlb(aux_var, new_lb)
            safe_setub(aux_var, new_ub)

            if isinstance(relaxation, PWXSquaredRelaxationData):
                # w >= x^2: add an oa point in the midpoint
                relaxation.clean_oa_points()
                var_values = pe.ComponentMap()
                x_var = relaxation.get_rhs_vars()[0]
                midpoint = x_var.lb + 0.5*(x_var.ub - x_var.lb)
                var_values[x_var] = midpoint
                relaxation.add_oa_point(var_values)

            relaxation.rebuild()

    def branch_at_point(self, branching_point, mc):
        assert self.branching_point is None
        self.branching_point = branching_point

        children_bounds = branch_at_point(
            self.root._model, self._bounds, branching_point, mc
        )

        return [
            NodeStorage(self.root, self, bounds, branching_point.variable)
            for bounds in children_bounds
        ]


class NodeStorage(_NodeStorageBase):
    def __init__(self, root, parent, bounds, branching_variable):
        super().__init__(root, parent, bounds)
        self.cut_pool = parent.cut_pool
        self.cut_node_storage = \
            CutNodeStorage(parent.cut_node_storage, parent.cut_pool)
        self.branching_variable = branching_variable

    @property
    def is_root(self):
        return False

    @property
    def model_to_relaxation_var_map(self):
        return self.root.model_to_relaxation_var_map


class RootNodeStorage(_NodeStorageBase):
    def __init__(self, model, relaxation):
        bounds = pe.ComponentMap(
            (var, var.bounds)
            for var in model.component_data_objects(pe.Var, active=True)
        )
        super().__init__(root=self, parent=None, bounds=bounds)
        self._model = model
        self._relaxation = relaxation

        # Lazily instantiate when creating linear model
        self._linear_model = None
        self._relaxation_data = None
        self.model_to_relaxation_var_map = None
        self.cut_pool = None
        self.cut_node_storage = None

    def model(self):
        return self._model

    def model_relaxation(self):
        if self._linear_model is not None:
            assert self.cut_pool is not None
            assert self.cut_node_storage is not None
            return self._linear_model

        self._relaxation_data = RelaxationData(self._model)
        self._linear_model = self._relaxation.relax(self._model, self._relaxation_data)
        self.model_to_relaxation_var_map = self._relaxation_data.original_to_new_var_map

        # Pre-compute nonlinear relaxations objects
        # This is also needed to avoid branching on auxiliary variables
        # introduced by cut generators.
        self._linear_model.galini_nonlinear_relaxations = \
            list(relaxation_data_objects(self._linear_model))
        self.cut_pool = CutPool(self._linear_model)
        self.cut_node_storage = CutNodeStorage(None, self.cut_pool)

        return self._linear_model

    @property
    def is_root(self):
        return True
