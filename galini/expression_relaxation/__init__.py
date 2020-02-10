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

"""Module containing expression_relaxation interfaces and implementations."""

__all__ = [
    'ExpressionRelaxation', 'RelaxationSide',
    'ExpressionRelaxationResult', 'McCormickExpressionRelaxation',
    'LinearExpressionRelaxation', 'UnivariateConcaveExpressionRelaxation',
    'SumOfUnderestimators', 'DisaggregateBilinearExpressionRelaxation',
]


from galini.expression_relaxation.expression_relaxation import (
    ExpressionRelaxation,
    RelaxationSide,
    ExpressionRelaxationResult
)
from galini.expression_relaxation.bilinear import McCormickExpressionRelaxation
from galini.expression_relaxation.linear import LinearExpressionRelaxation
from galini.expression_relaxation.concave import UnivariateConcaveExpressionRelaxation
from galini.expression_relaxation.composite import SumOfUnderestimators
from galini.expression_relaxation.disaggregate_bilinear import DisaggregateBilinearExpressionRelaxation
