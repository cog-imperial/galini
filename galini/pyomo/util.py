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

"""Utility functions for galini.pyomo."""

import pyomo.environ as pe



def safe_setlb(var, lb):
    if lb is not None:
        lb = float(lb)
    var.setlb(lb)


def safe_setub(var, ub):
    if ub is not None:
        ub = float(ub)
    var.setub(ub)


def model_variables(model):
    """Return a list of variables in the model"""
    for variables in model.component_map(pe.Var, active=True).itervalues():
        for idx in variables:
            yield variables[idx]


def model_constraints(model):
    """Return a list of constraints in the model"""
    for cons in model.component_map(pe.Constraint, active=True).itervalues():
        for idx in cons:
            yield cons[idx]


def model_objectives(model):
    """Return a list of objectives in the model"""
    for obj in model.component_map(pe.Objective, active=True).itervalues():
        for idx in obj:
            yield obj[idx]


class _ChainedComponentMap:
    def __init__(self, source, target):
        self._s = source
        self._t = target

    def __getitem__(self, item):
        v = self._s[item]
        return self._t[v]

    def get(self, item, default=None):
        v = self._s.get(item, None)
        if v is None:
            return default
        return self._t.get(v, default)


def chain_component_maps(source, target):
    return _ChainedComponentMap(source, target)
