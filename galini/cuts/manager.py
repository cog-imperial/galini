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

from galini.config.options import OptionsGroup, StringListOption
from galini.logging import get_logger
from galini.registry import Registry
from galini.timelimit import current_time, seconds_elapsed_since

logger = get_logger(__name__)


class CutsGeneratorsRegistry(Registry):
    """Registry of CutsGenerators."""

    def group_name(self):
        return 'galini.cuts_generators'


class _CutGeneratorCounter:
    def __init__(self, telemetry, name):
        self._cut_count = telemetry.create_counter('{}.cuts_count'.format(name))
        self._cut_time = telemetry.create_counter('{}.time'.format(name))

    def increment(self, cuts_count, time):
        self._cut_count.increment(cuts_count)
        self._cut_time.increment(time)


class CutsGeneratorsManager:
    """Manages cuts generators."""

    def __init__(self, galini):
        self.galini = galini
        self._cuts_counters = []
        self._generators = self._initialize_generators(galini)

    def _initialize_generators(self, galini):
        generators = []
        cuts_gen_config = galini.get_configuration_group('cuts_generator')

        for cut_gen_name in cuts_gen_config.generators:
            generator_cls = galini.get_cuts_generator(cut_gen_name)
            if generator_cls is None:
                # pylint: disable=line-too-long
                raise ValueError(
                    'Invalid cuts generator "{}", available cuts generators: {}'.format(
                        cut_gen_name,
                        ', '.join(galini.available_cuts_generators()),
                    ))
            generator_config = cuts_gen_config[generator_cls.name]
            generators.append(generator_cls(galini, generator_config))
            counter_name = "{}.cuts_count".format(generator_cls.name)
            self._cuts_counters.append(
                _CutGeneratorCounter(self.galini.telemetry, counter_name)
            )
        return generators

    @staticmethod
    def cuts_generators_manager_options():
        """Options for CutsGeneratorsManager."""
        return OptionsGroup('cuts_generator', [
            StringListOption('generators', default=[]),
        ])

    @property
    def generators(self):
        """Available cuts generators."""
        return self._generators

    def before_start_at_root(self, run_id, problem, relaxed_problem):
        """Callback called before start at root node."""
        for gen in self._generators:
            gen.before_start_at_root(run_id, problem, relaxed_problem)

    def after_end_at_root(self, run_id, problem, relaxed_problem, solution):
        """Callback called after end at root node."""
        for gen in self._generators:
            gen.after_end_at_root(run_id, problem, relaxed_problem, solution)

    def before_start_at_node(self, run_id, problem, relaxation):
        """Callback called before start at non root nodes."""
        for gen in self._generators:
            gen.before_start_at_node(run_id, problem, relaxation)

    def after_end_at_node(self, run_id, problem, relaxed_problem, solution):
        """Callback called after end at non root nodes."""
        for gen in self._generators:
            gen.after_end_at_node(run_id, problem, relaxed_problem, solution)

    def has_converged(self, state):
        """Predicated to check if cuts have converged."""
        return all(gen.has_converged(state) for gen in self._generators)

    def generate(self, run_id, problem, relaxed_problem, linear_problem,
                 mip_solution, tree, node):
        """Generate a new set of cuts."""
        all_cuts = []
        logger.info(
            run_id,
            'Generating cuts: {}',
            [gen.name for gen in self._generators],
        )

        paranoid_mode = self.galini.paranoid_mode

        for gen, counter in zip(self._generators, self._cuts_counters):
            start_time = current_time()
            cuts = gen.generate(
                run_id, problem, relaxed_problem, linear_problem, mip_solution,
                tree, node
            )
            elapsed_time = seconds_elapsed_since(start_time)

            if cuts is None:
                cuts = []

            if not isinstance(cuts, list):
                raise ValueError(
                    'CutsGenerator.generate must return a list of cuts.'
                )
            logger.info(
                run_id, '  * {} generated {} cuts.', gen.name, len(cuts)
            )

            for cut in cuts:
                if paranoid_mode:
                    _check_cut_coefficients(cut)
                all_cuts.append(cut)
            counter.increment(len(cuts), elapsed_time)
        return all_cuts


def _check_cut_coefficients(cut):
    raise NotImplemented()
