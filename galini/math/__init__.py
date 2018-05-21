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
"""Set GALINI math mode."""

class MathMode(object):
    """Math mode used internally by GALINI."""
    ARBITRARY_PRECISION = 1


# pylint: disable=undefined-all-variable
_COMMON_MEMBERS = [
    'inf',
    'pi',
    'sin',
    'cos',
    'sqrt',
    'log',
    'exp',
    'sin',
    'asin',
    'cos',
    'acos',
    'tan',
    'atan',
    'isnan',
    'almosteq',
    'almostgte',
    'almostlte',
]

_ARBITRARY_PRECISION_MEMBERS = [
    'mpf',
]

__all__ = _COMMON_MEMBERS


def set_math_mode(math_mode: int) -> None:
    """Set the math mode used by GALINI.

    Parameters
    ----------
    math_mode: MathMode
        the math mode to use
    """
    if math_mode == MathMode.ARBITRARY_PRECISION:
        from galini.math import arbitrary_precision as arb
        for member in _COMMON_MEMBERS:
            globals()[member] = getattr(arb, member)
        for member in _ARBITRARY_PRECISION_MEMBERS:
            globals()[member] = getattr(arb, member)
    else:
        raise RuntimeError('Invalid MathMode')


set_math_mode(MathMode.ARBITRARY_PRECISION)
