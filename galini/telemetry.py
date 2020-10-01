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
from contextlib import contextmanager
from galini.timelimit import seconds_elapsed_since, current_time


class Telemetry:
    def __init__(self, galini):
        self._logger = galini.get_logger(__name__)
        self._counters = dict()

    def create_counter(self, name, initial_value=0):
        if name in self._counters:
            raise ValueError('Counter {} already exists'.format(name))
        counter = Counter(name, initial_value)
        self._counters[name] = counter
        return counter

    def create_gauge(self, name, initial_value=0):
        if name in self._counters:
            raise ValueError('Counter {} already exists'.format(name))
        gauge = Gauge(name, initial_value)
        self._counters[name] = gauge
        return gauge

    def counters_values(self):
        def _counter_to_dict(counter):
            return {
                'name': counter.name,
                'value': counter.value,
            }
        return [
            _counter_to_dict(counter)
            for counter in self._counters.values()
        ]

    def log_at_end_of_iteration(self, iteration):
        self._logger.info(
            'Counters value at end of iteration {}',
            iteration,
        )
        for counter in self._counters.values():
            self._logger.update_variable(
                iteration=iteration,
                var_name=counter.name,
                value=counter.value,
            )
            self._logger.info(
                ' * {} = {}',
                counter.name,
                counter.value
            )

    def get_counter(self, name):
        return self._counters.get(name, None)

    def timespan(self, name):
        return timespan(self, name)


class Counter:
    def __init__(self, name, initial_value=0):
        self.name = name
        self._value = initial_value

    def increment(self, amount=1):
        self._value += amount
        return self._value

    def decrement(self, amount=1):
        self._value -= amount
        return self._value

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


@contextmanager
def timespan(telemetry, name):
    name = 'time.{}'.format(name)
    counter = telemetry.get_counter(name)
    if counter is None:
        counter = telemetry.create_counter(name, 0.0)

    telemetry._logger.log_solve_start(name)
    start = current_time()
    yield
    duration = seconds_elapsed_since(start)
    telemetry._logger.log_solve_end(name)
    counter.increment(duration)
