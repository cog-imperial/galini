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


class CliCommand(metaclass=abc.ABCMeta): # pragma: no cover
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
