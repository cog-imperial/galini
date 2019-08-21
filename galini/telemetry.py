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

"""Collect telemetry information from solver parts."""

from galini.logging import get_logger


class Telemetry:
    def __init__(self):
        self._logger = get_logger("galini.telemetry")
        self._counters = []

    def create_counter(self, name, initial_value=0):
        counter = Counter(name, initial_value)
        self._counters.append(counter)
        return counter

    def create_gauge(self, name, initial_value=0):
        gauge = Gauge(name, initial_value)
        self._counters.append(gauge)
        return gauge

    def log_at_end_of_iteration(self, run_id, iteration):
        self._logger.info(
            run_id,
            'Counters value at end of iteration {}',
            iteration,
        )
        for counter in self._counters:
            self._logger.update_variable(
                run_id,
                iteration=iteration,
                var_name=counter.name,
                value=counter.value,
            )
            self._logger.info(
                run_id,
                ' * {} = {}',
                counter.name,
                counter.value
            )


class Counter:
    def __init__(self, name, initial_value=0):
        self.name = name
        self._value = initial_value

    def increment(self, amount=1):
        self._value += amount

    def decrement(self, amount=1):
        self._value -= amount

    @property
    def value(self):
        return self._value


class Gauge:
    def __init__(self, name, initial_value=0):
        self.name = name
        self._value = initial_value

    def set_value(self, value):
        self._value = value

    @property
    def value(self):
        return self._value
