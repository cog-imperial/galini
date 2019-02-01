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
"""Compute quantities such as absolute and relative gap."""
import numpy as np


_finfo = np.finfo(np.float64)


def absolute_gap(ub, lb):
    """Compute absolute gap `ub - lb`."""
    return ub - lb


def relative_gap(ub, lb):
    """Compute relative gap `(ub - lb) / ub`."""
    if np.isclose(ub, 0):
        return (ub - lb) / _finfo.eps
    return (ub - lb) / np.abs(ub)
