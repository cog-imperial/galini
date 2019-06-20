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
"""Cuts generators manager."""
from galini.registry import Registry
from galini.logging import get_logger
from galini.config.options import OptionsGroup, StringListOption


logger = get_logger(__name__)


class CutsGeneratorsRegistry(Registry):
    """Registry of CutsGenerators."""
    def group_name(self):
        return 'galini.cuts_generators'


class CutsGeneratorsManager(object):
    def __init__(self, galini):
        self._generators = self._initialize_generators(galini)

    def _initialize_generators(self, galini):
        generators = []
        cuts_gen_config = galini.get_configuration_group('cuts_generator')

        for cut_gen_name in cuts_gen_config.generators:
            generator_cls = galini.get_cuts_generator(cut_gen_name)
            if generator_cls is None:
                raise ValueError(
                    'Invalid cuts generator "{}", available cuts generators: {}'.format(
                        cut_gen_name,
                        ', '.join(registry.keys())
                    ))
            generator_config = cuts_gen_config[generator_cls.name]
            generators.append(generator_cls(galini, generator_config))
        return generators

    @staticmethod
    def cuts_generators_manager_options():
        return OptionsGroup('cuts_generator', [
            StringListOption('generators', default=[]),
        ])

    @property
    def generators(self):
        return self._generators

    def before_start_at_root(self, run_id, problem, relaxed_problem):
        for gen in self._generators:
            gen.before_start_at_root(run_id, problem, relaxed_problem)

    def after_end_at_root(self, run_id, problem, relaxed_problem, solution):
        for gen in self._generators:
            gen.after_end_at_root(run_id, problem, relaxed_problem, solution)

    def before_start_at_node(self, run_id, problem, relaxation):
        for gen in self._generators:
            gen.before_start_at_node(run_id, problem, relaxation)

    def after_end_at_node(self, run_id, problem, relaxed_problem, solution):
        for gen in self._generators:
            gen.after_end_at_node(run_id, problem, relaxed_problem, solution)

    def generate(self, run_id, problem, relaxed_problem, linear_problem, mip_solution, tree, node):
        all_cuts = []
        logger.info(run_id, 'Generating cuts')

        for gen in self._generators:
            cuts = gen.generate(run_id, problem, relaxed_problem, linear_problem, mip_solution, tree, node)
            if cuts is None:
                cuts = []
            if not isinstance(cuts, list):
                raise ValueError('CutsGenerator.generate must return a list of cuts.')
            logger.info(run_id, '  * {} generated {} cuts.', gen.name, len(cuts))
            for cut in cuts:
                all_cuts.append(cut)
        return all_cuts
