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

from pyomo.core.base.expr import set_expression_tree_format
import pyomo.core.base.expr_common as common


def set_pyomo4_expression_tree() -> None:
    """Set Pyomo expression tree format to ``Mode.pyomo4_trees``.

    GALINI does not work with Pyomo default tree format, so this
    function should be called at the beginning of every program.
    """
    set_expression_tree_format(common.Mode.pyomo4_trees)
