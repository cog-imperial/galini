# Copyright 2019 Francesco Ceccon
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

"""Factor nonlinear expressions in simpler expressions."""

from suspect.expression import ExpressionType
from galini.core import (
    LinearExpression,
    SumExpression,
    Constraint,
    Domain,
    ExpressionReference,
)
from galini.transformation import Transformation, TransformationResult


class FactorTransformation(Transformation):
    def __init__(self, source, target):
        super().__init__(source, target)
        self._memo = {}

    """Replace Nonlinear expressions with auxiliary variables."""
    def apply(self, expr, ctx):
        assert expr.problem is None or expr.problem == self.source
