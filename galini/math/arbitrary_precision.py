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

# pylint: disable=invalid-name
"""Arbitrary precision mathematical constants and comparison."""
from typing import Any
import mpmath


mpf = mpmath.mpf
inf = mpf('inf')
pi = mpmath.pi
sin = mpmath.sin
cos = mpmath.cos
sqrt = mpmath.sqrt
log = mpmath.log
exp = mpmath.exp
sin = mpmath.sin
asin = mpmath.asin
cos = mpmath.cos
acos = mpmath.acos
tan = mpmath.tan
atan = mpmath.atan
isnan = mpmath.isnan


def almosteq(a: Any, b: Any) -> bool:
    """Floating point equality check between `a` and `b`."""
    # in mpmath inf != inf, but we want inf == inf
    if abs(a) == inf and abs(b) == inf:
        return (a > 0 and b > 0) or (a < 0 and b < 0)
    return mpmath.almosteq(a, b)


def almostgte(a: Any, b: Any) -> bool:
    """Return True if a >= b."""
    return a > b or almosteq(a, b)


def almostlte(a: Any, b: Any) -> bool:
    """Return True if a <= b."""
    return a < b or almosteq(a, b)
