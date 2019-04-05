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
"""GALINI CLI entry point."""

import sys
import argparse
import pkg_resources


def collect_commands(parser, subcommands_entry_points_iter):
    """Collect all commands registered with `galini.commands`.

    Returns
    -------
    dict
        a dict of command names and objects.
    """
    commands = {}
    for entry_point in subcommands_entry_points_iter:
        if entry_point.name in commands:
            print('Duplicate entry point {} found.'.format(entry_point.name))
            sys.exit(1)
        sub_cls = entry_point.load()
        sub = sub_cls()
        subparser = parser.add_parser(entry_point.name, help=sub.help_message())
        sub.add_parser_arguments(subparser)
        commands[entry_point.name] = sub

    return commands


def main(): # pragma: no cover
    """Main entry point."""
    parser = argparse.ArgumentParser(prog='galini')
    subparser = parser.add_subparsers(dest='command')
    subcommands = collect_commands(
        subparser,
        pkg_resources.iter_entry_points('galini.commands'),
    )
    args = parser.parse_args()

    if args.command is None:
        parser.print_help(sys.stderr)
        sys.exit(1)

    command = subcommands.get(args.command)

    if command is None:
        print('Invalid command {}'.format(command))
        sys.exit(1)

    command.execute(args)
