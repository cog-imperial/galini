#  Copyright 2019 Francesco Ceccon
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.

"""Slsqp solver solution."""
import numpy as np
from galini.logging import get_logger
from galini.math import mc, almost_ge, almost_le
from galini.core import (
    IpoptSolution as CoreIpoptSolution,
)
from galini.pyomo.postprocess import PROBLEM_RLT_CONS_INFO
from galini.solvers import (
    Solution,
    Status,
    OptimalObjective,
    OptimalVariable,
)


class SlsqpSolutionSuccess(Status):
    def __init__(self, message):
        self.message = message

    def is_success(self):
        return True

    def is_infeasible(self):
        return False

    def is_unbounded(self):
        return False

    def description(self):
        return 'Success: {}'.format(self.message)


class SlsqpSolutionFailure(Status):
    def __init__(self, message):
        self.message = message

    def is_success(self):
        return False

    def is_infeasible(self):
        return True

    def is_unbounded(self):
        return False

    def description(self):
        return 'Failure: {}'.format(self.message)


class SlsqpSolution(Solution):
    def __str__(self):
        return 'SlsqpSolution(status={}, objective={})'.format(
            self.status.description(),
            self.objective,
        )

    def __repr__(self):
        return '<{} at {}>'.format(str(self), hex(id(self)))
