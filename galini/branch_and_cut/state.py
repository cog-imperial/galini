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

"""Cut loop state."""
import numpy as np

from galini.math import is_close


class CutsState:
    """Cut loop state."""
    def __init__(self):
        self.round = 0
        self.lower_bound = -np.inf
        self.first_solution = None
        self.latest_solution = None
        self.previous_solution = None

    def update(self, solution, paranoid=False, atol=None, rtol=None):
        """Update cut state with `solution`."""
        self.round += 1
        current_objective = solution.objective
        if paranoid:
            close = is_close(
                current_objective, self.lower_bound, atol=atol, rtol=rtol
            )
            increased = (
                current_objective >= self.lower_bound or
                close
            )
            if not increased:
                msg = 'Lower bound in cuts phase decreased: {} to {}'
                raise RuntimeError(
                    msg.format(self.lower_bound, current_objective)
                )

        self.lower_bound = current_objective
        if self.first_solution is None:
            self.first_solution = current_objective
        self.previous_solution = self.latest_solution
        self.latest_solution = current_objective

    def __str__(self):
        return 'CutsState(round={}, lower_bound={})'.format(
            self.round, self.lower_bound
        )

    def __repr__(self):
        return '<{} at {}>'.format(str(self), hex(id(self)))
