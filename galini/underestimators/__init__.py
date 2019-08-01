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

"""Module containing underestimators interfaces and implementations."""

__all__ = [
    'Underestimator', 'UnderestimatorResult', 'McCormickUnderestimator',
    'LinearUnderestimator', 'UnivariateConcaveUnderestimator',
    'SumOfUnderestimators', 'DisaggregateBilinearUnderestimator',
]


from galini.underestimators.underestimator import (
    Underestimator,
    UnderestimatorResult
)
from galini.underestimators.bilinear import McCormickUnderestimator
from galini.underestimators.linear import LinearUnderestimator
from galini.underestimators.concave import UnivariateConcaveUnderestimator
from galini.underestimators.composite import SumOfUnderestimators
from galini.underestimators.disaggregate_bilinear import DisaggregateBilinearUnderestimator
