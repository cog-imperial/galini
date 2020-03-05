# Copyright 2020 Francesco Ceccon
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

from coramin.relaxations.auto_relax import (
    relax as coramin_relax,
    _relax_root_to_leaf_map,
    _relax_leaf_to_root_map,
    _relax_leaf_to_root_SumExpression,
    _relax_root_to_leaf_SumExpression,
)
from suspect.pyomo.quadratic import QuadraticExpression


_relax_leaf_to_root_map[QuadraticExpression] = _relax_leaf_to_root_SumExpression
_relax_root_to_leaf_map[QuadraticExpression] = _relax_root_to_leaf_SumExpression


def relax(model):
    return coramin_relax(model, in_place=False, use_fbbt=False)