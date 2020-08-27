#  Copyright 2020 Francesco Ceccon
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

"""Branch & Cut telemetry.

References
----------

Berthold, T. (2013). Measuring the impact of primal heuristics.
    Operations Research Letters, 41(6), 611–614. https://doi.org/10.1016/J.ORL.2013.08.007
"""

import numpy as np
from galini.math import is_close, is_inf
from galini.quantities import relative_gap


def update_gauge(galini, name, value, initial_value=0.0):
    """Update a gauge counter, creating the gauge if it does not exists."""
    telemetry = galini.telemetry
    counter = telemetry.get_counter(name)
    if counter is None:
        counter = telemetry.create_gauge(name, initial_value)
    counter.set_value(value)


def update_counter(galini, name, amount, initial_value=0.0):
    """Update a counter, creating the gauge if it does not exists."""
    telemetry = galini.telemetry
    counter = telemetry.get_counter(name)
    if counter is None:
        counter = telemetry.create_counter(name, initial_value)
    return counter.increment(amount)


def increment_nodes_visited(galini):
    """Increment the number of nodes visited by 1."""
    return update_counter(galini, 'branch_and_bound.nodes_visited', 1, initial_value=0)


def increment_elapsed_time(galini):
    """Increment the elapsed time gauge."""
    return update_gauge(galini, 'elapsed_time', galini.timelimit.elapsed_time(), initial_value=0.0)


def update_solution_bound(galini, tree, delta_t):
    """Increment the lower and upper bound and their integrals."""
    update_gauge(galini, 'branch_and_bound.lower_bound', tree.lower_bound)
    update_gauge(galini, 'branch_and_bound.upper_bound', tree.upper_bound)

    # Compute relative gap
    gap = relative_gap(tree.lower_bound, tree.upper_bound, galini.mc)
    gap = np.min([gap, 1.0])
    update_gauge(galini, 'branch_and_bound.relative_gap', gap, initial_value=1.0)
    update_counter(galini, 'branch_and_bound.relative_gap_integral', gap * delta_t, initial_value=0.0)


def update_at_end_of_iteration(galini, tree, delta_t):
    """Update all bab counters at the end of one iteration."""
    bab_iteration = increment_nodes_visited(galini)
    update_solution_bound(galini, tree, delta_t)
    increment_elapsed_time(galini)
    # finally, output all counters
    galini.telemetry.log_at_end_of_iteration(bab_iteration)
