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

"""GALINI Math context."""
import numpy as np


class MathContext:
    def __init__(self):
        self.epsilon = 1e-5
        self.infinity = 1e20
        self.integer_infinity = 2**63 - 1
        self.constraint_violation_tol = 1e-8


mc = MathContext()


def is_close(a, b, atol=None, rtol=None):
    if atol is None and rtol is None:
        raise ValueError('One of atol and rtol must be specified')
    if atol is None:
        atol = 0.0
    if rtol is None:
        rtol = 0.0

    return np.isclose(a, b, atol=atol, rtol=rtol)


def is_inf(n):
    return (
        np.isinf(n) or
        n >= mc.infinity or
        n <= -mc.infinity
    )


def almost_ge(a, b, atol=None, rtol=None):
    if atol is None and rtol is None:
        raise ValueError('One of atol and rtol must be specified')
    if atol is None:
        atol = 0.0
    if rtol is None:
        rtol = 0.0

    if a > b:
        return True
    if np.isclose(a, b, atol=atol, rtol=rtol):
        return True
    return False


def almost_le(a, b, atol=None, rtol=None):
    if atol is None and rtol is None:
        raise ValueError('One of atol and rtol must be specified')
    if atol is None:
        atol = 0.0
    if rtol is None:
        rtol = 0.0

    if a < b:
        return True
    if np.isclose(a, b, atol=atol, rtol=rtol):
        return True
    return False
