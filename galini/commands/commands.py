# Copyright 2018 Francesco Ceccon
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

"""CLI commands base class."""

import abc

import pyomo.environ as pe
from suspect.pyomo.connected_model import create_connected_model

from galini.pyomo import read_pyomo_model


class CliCommand(metaclass=abc.ABCMeta):  # pragma: no cover
    """Abstract class for CLI commands."""
    @abc.abstractmethod
    def execute(self, args):
        """Run the command."""
        pass

    @abc.abstractmethod
    def help_message(self):
        """Return the command help message."""
        pass

    @abc.abstractmethod
    def add_parser_arguments(self, parser):
        """Add arguments specific to this command to the argument parser."""
        pass


class CliCommandWithProblem(CliCommand):
    """A CLI Command that receives a problem as first argument"""
    def execute(self, args):
        assert args.problem
        pyomo_model = read_pyomo_model(
            args.problem,
            objective_prefix=args.objective_prefix,
        )
        connected_pyomo_model, _ = create_connected_model(pyomo_model)
        return self.execute_with_model(connected_pyomo_model, args)

    @abc.abstractmethod
    def execute_with_model(self, model: pe.ConcreteModel, args):
        """Run the command."""
        pass

    def add_parser_arguments(self, parser):
        parser.add_argument('problem')
        parser.add_argument('--objective-prefix')
        self.add_extra_parser_arguments(parser)

    @abc.abstractmethod
    def add_extra_parser_arguments(self, parser):
        """Add extra arguments to this command parser."""
        pass
