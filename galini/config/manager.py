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

"""Configuration Manager."""

import toml

from galini.config.configuration import GaliniConfig
from galini.config.options import (
    OptionsGroup,
    ExternalSolverOptions,
    EnumOption,
    NumericOption,
    IntegerOption,
    StringOption,
    BoolOption,
)
from galini.cuts import CutsGeneratorsManager


class ConfigurationManager(object):
    def __init__(self, algos_reg, cuts_gen_reg):
        self._initialized = False
        self._configuration = None
        self._initialize(algos_reg, cuts_gen_reg)

    def _initialize(self, algos_reg, cuts_gen_reg):
        config = GaliniConfig()

        # add default sections
        logging_group = config.add_group('logging')
        _assign_options_to_group(_logging_group(), logging_group)

        galini_group = config.add_group('galini')
        _assign_options_to_group(_galini_group(), galini_group)

        # initialize configuration from solvers
        for _, algo in algos_reg.items():
            algo_options = algo.algorithm_options()
            group = config.add_group(algo_options.name)
            _assign_options_to_group(algo_options, group)

        # initialize configuration from cut manager and generators
        cuts_generator_group = config.add_group('cuts_generator')
        _assign_options_to_group(
            CutsGeneratorsManager.cuts_generators_manager_options(),
            cuts_generator_group,
        )
        for _, generator in cuts_gen_reg.items():
            cuts_gen_options = generator.cuts_generator_options()
            group = cuts_generator_group.add_group(cuts_gen_options.name)
            _assign_options_to_group(cuts_gen_options, group)

        self._configuration = config
        self._initialized = True

    def update_configuration(self, user_config):
        if not isinstance(user_config, dict):
            user_config = toml.load(user_config)
        self._configuration.update(user_config)

    @property
    def configuration(self):
        if not self._initialized:
            raise RuntimeError('ConfigurationManager was not initialized.')
        return self._configuration


def _logging_group():
    return OptionsGroup('logging', [
        EnumOption('level', ['NOTSET', 'DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'], 'INFO'),
        BoolOption('stdout', default=True),
        StringOption('directory', default=None),
    ])


def _galini_group():
    return OptionsGroup('galini', [
        NumericOption('timelimit', min_value=0, default=86400),
        NumericOption('infinity', min_value=0, default=1e20),
        NumericOption('integer_infinity', min_value=0, default=2**63 - 1),
        NumericOption('user_upper_bound', min_value=0, default=1e9),
        NumericOption('user_integer_upper_bound', min_value=0, default=1e5),
        NumericOption('epsilon', min_value=0, default=1e-6),
        NumericOption('constraint_violation_tol', min_value=0, default=1e-6),
        IntegerOption('fbbt_quadratic_max_terms', min_value=1, default=30),
        IntegerOption('fbbt_sum_max_children', min_value=1, default=30),
        IntegerOption('fbbt_linear_max_children', min_value=1, default=30),
        IntegerOption('special_structure_quadratic_max_terms', min_value=1, default=500),
        IntegerOption('special_structure_sum_max_children', min_value=1, default=500),
        IntegerOption('special_structure_linear_max_children', min_value=1, default=500),
        BoolOption('paranoid_mode', default=False),
    ])


def _assign_options_to_group(options, group):
    for option in options.iter():
        if isinstance(option, ExternalSolverOptions):
            group.add_group(option.name, strict=False)
        elif isinstance(option, OptionsGroup):
            sub_group = group.add_group(option.name)
            _assign_options_to_group(option, sub_group)
        else:
            group.set(option.name, option.default)
