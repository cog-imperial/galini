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

"""Branch & Cut telemetry."""

import numpy as np
from galini.math import is_close, is_inf, mc


class PrimalDualIntegral:
    """Compute and track primal and dual integrals.

    References
    ----------

    Berthold, T. (2013).
    Measuring the impact of primal heuristics.
    Operations Research Letters, 41(6), 611–614.
    https://doi.org/10.1016/J.ORL.2013.08.007
    """
    def __init__(self, telemetry):
        self._initialized = False
        self.optimal_objective = None
        self.previous_time = None
        self._upper_bound = \
            telemetry.create_counter('branch_and_cut.upper_bound_integral', 0)
        self._lower_bound = \
            telemetry.create_counter('branch_and_cut.lower_bound_integral', 0)
        self._upper_bound_pct = \
            telemetry.create_gauge('branch_and_cut.upper_bound_integral_pct', 1.0)
        self._lower_bound_pct = \
            telemetry.create_gauge('branch_and_cut.lower_bound_integral_pct', 1.0)

    def _compute_gamma(self, optimal_obj, value):
        # Unknown primal
        if value is None or is_inf(value):
            return 1.0

        # |opt| = |obj| = 0.0
        if is_close(np.abs(optimal_obj), 0.0, atol=mc.epsilon):
            if is_close(np.abs(value), 0.0, atol=mc.epsilon):
                return 0.0

        # opt * obj < 0
        if np.sign(optimal_obj) * np.sign(value) < 0:
            return 1.0

        num = np.abs(optimal_obj - value)
        den = max(np.abs(optimal_obj), np.abs(value))
        return num / den

    def start_timing(self, optimal_objective, elapsed_time):
        self.optimal_objective = optimal_objective
        self.previous_time = elapsed_time
        self._initialized = True

    def update_at_end_of_iteration(self, tree, elapsed_time):
        if not self._initialized:
            raise RuntimeError(
                'PrimalDualIntegral not initialized. Call start_timing'
            )
        if self.optimal_objective is None:
            return
        delta_t = elapsed_time - self.previous_time
        upper_bound_gamma = \
            self._compute_gamma(self.optimal_objective, tree.upper_bound)
        self._upper_bound_pct.set_value(upper_bound_gamma)
        self._upper_bound.increment(upper_bound_gamma * delta_t)
        lower_bound_gamma = \
            self._compute_gamma(self.optimal_objective, tree.lower_bound)
        self._lower_bound_pct.set_value(lower_bound_gamma)
        self._lower_bound.increment(lower_bound_gamma * delta_t)

class BranchAndCountTelemetry:
    """Collection of telemetry information of the Branch & Cut algorithm."""
    def __init__(self, telemetry):
        self._primal_dual = PrimalDualIntegral(telemetry)
        self._upper_bound = \
            telemetry.create_gauge('branch_and_cut.upper_bound', np.inf)
        self._lower_bound = \
            telemetry.create_gauge('branch_and_cut.lower_bound', -np.inf)
        self._nodes_visited = \
            telemetry.create_counter('branch_and_cut.nodes_visited', 0)
        self._total_cut_rounds = \
            telemetry.create_counter('branch_and_cut.total_cut_rounds', 0)
        self._inherited_cuts = \
            telemetry.create_counter('branch_and_cut.inherited_cuts')
        self._obbt_time = \
            telemetry.create_counter('branch_and_cut.obbt_time', 0)
        self._fbbt_time = \
            telemetry.create_counter('branch_and_cut.fbbt_time', 0)
        self._lower_bound_time = \
            telemetry.create_counter('branch_and_cut.lower_bound_time', 0)
        self._upper_bound_time = \
            telemetry.create_counter('branch_and_cut.upper_bound_time', 0)

    def start_timing(self, optimal_objective, elapsed_time):
        self._primal_dual.start_timing(optimal_objective, elapsed_time)

    def increment_inherited_cuts(self, count):
        """Increment inherited cuts counter."""
        self._inherited_cuts.increment(count)

    def increment_total_cut_rounds(self):
        """Increment total cut rounds by 1."""
        self._total_cut_rounds.increment(1)

    def increment_obbt_time(self, elapsed):
        """Increment OBBT elapsed time counter."""
        self._obbt_time.increment(elapsed)

    def increment_fbbt_time(self, elapsed):
        """Increment FBBT elapsed time counter."""
        self._fbbt_time.increment(elapsed)

    def increment_lower_bound_time(self, elapsed):
        """Increment lower bound search elapsed time counter."""
        self._lower_bound_time.increment(elapsed)

    def increment_upper_bound_time(self, elapsed):
        """Increment upper bound search elapsed time counter."""
        self._upper_bound_time.increment(elapsed)

    def update_at_end_of_iteration(self, tree, elapsed_time=None):
        """Update tree state counters at end of iteration."""
        self._primal_dual.update_at_end_of_iteration(tree, elapsed_time)
        self._upper_bound.set_value(tree.upper_bound)
        self._lower_bound.set_value(tree.lower_bound)
        self._nodes_visited.increment(1)
