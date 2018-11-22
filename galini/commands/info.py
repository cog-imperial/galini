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

"""GALINI info command."""
from texttable import Texttable
from galini.commands import (
    CliCommandWithProblem,
    OutputTable,
    add_output_format_parser_arguments,
    print_output_table,
)


class InfoCommand(CliCommandWithProblem):
    """Command to output information about a problem."""
    def execute_with_problem(self, problem, args):
        tables = []
        tables.append(self._output_variables_information(problem))
        tables.append(self._output_objectives_information(problem))
        tables.append(self._output_constraints_information(problem))
        print_output_table(tables, args)

    def _output_variables_information(self, problem):
        table = OutputTable('Variables', ['name', 'lower_bound', 'upper_bound'])
        for name, var in problem.variables.items():
            table.add_row({'name': name, 'lower_bound': var.lower_bound, 'upper_bound': var.upper_bound})
        return table

    def _output_objectives_information(self, problem):
        table = OutputTable('Objectives', ['name', 'sense'])
        for name, obj in problem.objectives.items():
            table.add_row({'name': name, 'sense': 'minimize'})
        return table

    def _output_constraints_information(self, problem):
        table = OutputTable('Constraints', ['name', 'lower_bound', 'upper_bound'])
        for name, cons in problem.constraints.items():
            table.add_row({'name': name, 'lower_bound': cons.lower_bound, 'upper_bound': cons.upper_bound})
        return table

    def help_message(self):
        return "Print information about the problem"

    def add_extra_parser_arguments(self, parser):
        add_output_format_parser_arguments(parser)