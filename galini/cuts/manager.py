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

import numpy as np
from pyomo.core.expr.visitor import StreamBasedExpressionVisitor
from pyomo.core.expr.current import nonpyomo_leaf_types

from galini.config.options import OptionsGroup, StringListOption
from galini.registry import Registry
from galini.timelimit import current_time, seconds_elapsed_since


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
        self.logger = galini.get_logger(__name__)
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

    def before_start_at_root(self, problem, relaxed_problem):
        """Callback called before start at root node."""
        for gen in self._generators:
            gen.before_start_at_root(problem, relaxed_problem)

    def after_end_at_root(self, problem, relaxed_problem, solution):
        """Callback called after end at root node."""
        for gen in self._generators:
            gen.after_end_at_root(problem, relaxed_problem, solution)

    def before_start_at_node(self, problem, relaxation):
        """Callback called before start at non root nodes."""
        for gen in self._generators:
            gen.before_start_at_node(problem, relaxation)

    def after_end_at_node(self, problem, relaxed_problem, solution):
        """Callback called after end at non root nodes."""
        for gen in self._generators:
            gen.after_end_at_node(problem, relaxed_problem, solution)

    def has_converged(self, state):
        """Predicated to check if cuts have converged."""
        return all(gen.has_converged(state) for gen in self._generators)

    def generate(self, problem, relaxed_problem, mip_solution, tree, node):
        """Generate a new set of cuts."""
        all_cuts = []
        self.logger.info(
            'Generating cuts: {}',
            [gen.name for gen in self._generators],
        )

        for gen, counter in zip(self._generators, self._cuts_counters):
            start_time = current_time()
            cuts = gen.generate(
                problem, relaxed_problem, mip_solution, tree, node
            )
            elapsed_time = seconds_elapsed_since(start_time)

            if cuts is None:
                cuts = []

            if not isinstance(cuts, list):
                raise ValueError(
                    'CutsGenerator.generate must return a list of cuts.'
                )
            self.logger.info(
                '  * {} generated {} cuts.', gen.name, len(cuts)
            )

            for cut in cuts:
                if not self.galini.debug_assert_(
                        lambda: _check_cut_coefficients_are_numerically_reasonable(cut),
                        'Numerical coefficients in cut are not reasonable'):
                    from galini.ipython import embed_ipython
                    embed_ipython(header='Numerical coefficients in cut are not reasonable')

                all_cuts.append(cut)
            counter.increment(len(cuts), elapsed_time)
        return all_cuts


def _check_cut_coefficients_are_numerically_reasonable(cut):
    def enter_node(node):
        return None, []

    def exit_node(node, data):
        node_type = type(node)
        if node_type in nonpyomo_leaf_types:
            return np.isfinite(node)
        elif not node.is_expression_type() and not node.is_variable_type():
            return np.isfinite(node.value)
        return True

    return StreamBasedExpressionVisitor(
        enterNode=enter_node,
        exitNode=exit_node,
    ).walk_expression(cut)

