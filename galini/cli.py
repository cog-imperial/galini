"""GALINI CLI entry point."""
import sys
import logging
import argparse
import pkg_resources
from galini.pyomo import set_pyomo4_expression_tree


def collect_subcommands(parser):
    """Collect all subcommands registered with `galini.subcommands`.

    Returns
    -------
    dict
        a dict of command names and objects.
    """
    subcommands = {}
    for entry_point in pkg_resources.iter_entry_points('galini.subcommands'):
        if entry_point.name in subcommands:
            logging.error('Duplicate entry point %s found.', entry_point.name)
            sys.exit(1)
        sub_cls = entry_point.load()
        sub = sub_cls()
        subparser = parser.add_parser(entry_point.name, help=sub.help_message())
        sub.add_parser_arguments(subparser)
        subcommands[entry_point.name] = sub

    return subcommands


def main():
    """Main entry point."""
    set_pyomo4_expression_tree()

    parser = argparse.ArgumentParser(prog='galini')
    subparser = parser.add_subparsers(dest='command')
    subcommands = collect_subcommands(subparser)
    args = parser.parse_args()

    if args.command is None:
        parser.print_help(sys.stderr)
        sys.exit(1)

    command = subcommands.get(args.command)

    if command is None:
        logging.error('Invalid command %s', command)
        sys.exit(1)

    command.execute(args)
