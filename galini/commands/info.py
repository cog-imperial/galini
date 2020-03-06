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
import pyomo.environ as pe

from galini.commands import (
    CliCommandWithProblem,
    OutputTable,
    add_output_format_parser_arguments,
    print_output_table,
)


class InfoCommand(CliCommandWithProblem):
    """Command to output information about a problem."""
    def execute_with_model(self, model: pe.ConcreteModel, args):
        tables = [
            self._output_variables_information(model),
            self._output_objectives_information(model),
            self._output_constraints_information(model),
        ]
        print_output_table(tables, args)

    def _output_variables_information(self, model: pe.ConcreteModel):
        table = OutputTable('Variables', ['name', 'lower_bound', 'upper_bound', 'domain'])
        for var in model.component_data_objects(pe.Var, active=True, descend_into=True):
            table.add_row({
                'name': var.name,
                'lower_bound': var.lb,
                'upper_bound': var.ub,
                'domain': var.domain,
            })
        return table

    def _output_objectives_information(self, model: pe.ConcreteModel):
        table = OutputTable('Objectives', ['name', 'sense'])
        for obj in model.component_data_objects(pe.Objective, active=True, descend_into=True):
            table.add_row({'name': obj.name, 'sense': obj.sense})
        return table

    def _output_constraints_information(self, model: pe.ConcreteModel):
        table = OutputTable('Constraints', ['name', 'lower_bound', 'upper_bound'])
        for cons in model.component_data_objects(pe.Constraint, active=True, descend_into=True):
            table.add_row({
                'name': cons.name,
                'lower_bound': cons.lower,
                'upper_bound': cons.upper,
            })
        return table

    def help_message(self):
        return "Print information about the problem"

    def add_extra_parser_arguments(self, parser):
        add_output_format_parser_arguments(parser)
