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

"""GALINI plugins subcommand."""

import sys

from galini.commands import (
    CliCommand,
    OutputTable,
    print_output_table,
    add_output_format_parser_arguments,
)
from galini.cuts import CutsGeneratorsRegistry


class PluginsCommand(CliCommand):
    """Command to list registered plugins."""

    _methods = {
        'cuts': '_get_cuts',
    }

    def execute(self, args):
        method_name = self._methods.get(args.selection, None)
        method = getattr(self, method_name, None) if method_name is not None else None
        if method is None:
            print('Callback for selection {} not found.'.format(args.selection))
            print('This is a bug in GALINI. Please open an issue at https://github.com/cog-imperial/galini')
            sys.exit(1)
        result = method()
        print_output_table(result, args)
        print()

    def help_message(self):
        return "List registered plugins"

    def add_parser_arguments(self, parser):
        parser.add_argument('selection', choices=['cuts'])
        add_output_format_parser_arguments(parser)

    def _get_cuts(self):
        table = OutputTable('CutsGenerators', [
            {'id': 'id', 'name': 'ID', 'type': 't'},
            {'id': 'name', 'name': 'Name', 'type': 't'},
            {'id': 'description', 'name': 'Description', 'type': 't'}
        ])

        registry = CutsGeneratorsRegistry()

        for key, solver in registry.items():
            table.add_row({
                'id': key,
                'name': solver.name,
                'description': solver.description
            })

        return table
