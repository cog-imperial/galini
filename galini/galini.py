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
from galini.cuts import CutsGeneratorsRegistry, CutsGeneratorsManager
from galini.config import ConfigurationManager
from galini.solvers import SolversRegistry
from galini.logging import (
    get_logger,
    apply_config as apply_log_config,
)


class Galini(object):
    def __init__(self):
        self._solvers_reg = SolversRegistry()
        self._cuts_gen_reg = CutsGeneratorsRegistry()
        self._config_manager = ConfigurationManager(self._solvers_reg, self._cuts_gen_reg)
        self._config = self._config_manager.configuration
        self.logger = get_logger('galini')
        self.cuts_generators_manager = CutsGeneratorsManager(self)

    def update_configuration(self, user_config):
        self._config_manager.update_configuration(user_config)
        self._config = self._config_manager.configuration
        self.cuts_generators_manager = CutsGeneratorsManager(self)
        apply_log_config(self._config.logging)

    def get_solver(self, name):
        """Get solver from the registry."""
        solver_cls = self._solvers_reg.get(name, None)
        if solver_cls is None:
            raise ValueError('No solver "{}"'.format(name))
        return solver_cls

    def instantiate_solver(self, solver_name):
        """Get and instantiate solver from the registry."""
        solver_cls = self.get_solver(solver_name)
        return solver_cls(self)

    def get_cuts_generator(self, name):
        """Get cuts generator from the registry."""
        gen_cls = self._cuts_gen_reg.get(name, None)
        if gen_cls is None:
            raise ValueError('No cuts generator "{}"'.format(name))
        return gen_cls

    def get_configuration_group(self, group):
        parts = group.split('.')
        config = self._config
        for part in parts:
            config = config.get(part, None)
            if config is None:
                raise ValueError('Invalid configuration group "{}"'.format(group))
        return config
