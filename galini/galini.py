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

"""GALINI root object. Contains global state."""
from suspect.pyomo.quadratic import enable_standard_repn_for_quadratic_expression

from galini.algorithms import AlgorithmsRegistry
from galini.config import ConfigurationManager
from galini.cuts import CutsGeneratorsRegistry, CutsGeneratorsManager
from galini.io.logging import LogManager
from galini.math import MathContext
from galini.telemetry import Telemetry
from galini.timelimit import Timelimit


class Galini:
    """Contains information about the current instance of galini."""

    def __init__(self):
        self._algos_reg = AlgorithmsRegistry()
        self._cuts_gen_reg = CutsGeneratorsRegistry()
        self._log_manager = LogManager()
        self._config_manager = \
            ConfigurationManager(self._algos_reg, self._cuts_gen_reg)

        self._config = self._config_manager.configuration
        self._telemetry = Telemetry(self)

        self.timelimit = Timelimit(0)
        self.cuts_generators_manager = CutsGeneratorsManager(self)
        self.paranoid_mode = False
        self.mc = MathContext()
        self.logger = self._log_manager.get_logger('galini')
        enable_standard_repn_for_quadratic_expression()

    @property
    def config(self):
        return self.get_configuration_group('galini')

    def update_configuration(self, user_config):
        """Update galini configuration with `user_config`."""
        self._config_manager.update_configuration(user_config)
        self._config = self._config_manager.configuration
        self.cuts_generators_manager = CutsGeneratorsManager(self)
        self._log_manager.apply_config(self._config.logging)
        _update_math_context(self.mc, self.config)
        self.paranoid_mode = self.config['paranoid_mode']

    def assert_(self, func, msg):
        if not func():
            if self.paranoid_mode:
                return False
            else:
                raise AssertionError(msg)
        return True

    def debug_assert_(self, func, msg):
        if self.paranoid_mode:
            if not func():
                if self.paranoid_mode:
                    return False
        return True

    def get_logger(self, name):
        return self._log_manager.get_logger(name)

    def get_algorithm(self, name):
        """Get algorithm from the registry."""
        algos_cls = self._algos_reg.get(name, None)
        if algos_cls is None:
            raise ValueError('No algorithm "{}"'.format(name))
        return algos_cls

    def instantiate_algorithm(self, algo_name):
        """Get and instantiate an algorithm from the registry."""
        algo_cls = self.get_algorithm(algo_name)
        return algo_cls(self)

    def available_algorithms(self):
        """Get avariable algorithms."""
        return self._algos_reg.keys()

    def get_cuts_generator(self, name):
        """Get cuts generator from the registry."""
        gen_cls = self._cuts_gen_reg.get(name, None)
        if gen_cls is None:
            raise ValueError('No cuts generator "{}"'.format(name))
        return gen_cls

    def available_cuts_generators(self):
        """Get available cuts generators."""
        return self._cuts_gen_reg.keys()

    def get_configuration_group(self, group):
        """Get the specified configuration `group`."""
        parts = group.split('.')
        config = self._config
        for part in parts:
            config = config.get(part, None)
            if config is None:
                raise ValueError(
                    'Invalid configuration group "{}"'.format(group)
                )
        return config

    @property
    def telemetry(self):
        return self._telemetry


def _update_math_context(mc, galini):
    mc.epsilon = galini.epsilon
    mc.infinity = galini.infinity
    mc.integer_infinity = galini.integer_infinity
    mc.constraint_violation_tol = galini.constraint_violation_tol
    mc.user_upper_bound = galini.user_upper_bound
    mc.user_integer_upper_bound = galini.user_integer_upper_bound
