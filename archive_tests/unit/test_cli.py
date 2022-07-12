# pylint: skip-file
import pytest
import argparse
from galini.cli import collect_commands


class _MockCommand():
    def help_message(self):
        return 'help'

    def add_parser_arguments(self, parser):
        pass


class _MockEntryPoint:
    def __init__(self, name):
        self.name = name

    def load(self):
        return _MockCommand


def mock_entry_points_iter(entry_points):
    for name in entry_points:
        yield _MockEntryPoint(name)


def test_collect_subcommands():
    parser = argparse.ArgumentParser()
    subparser = parser.add_subparsers(dest='command')
    subcommands = collect_commands(
        subparser,
        mock_entry_points_iter(['one', 'two', 'three']),
    )
    assert 'one' in subcommands
    assert 'two' in subcommands
    assert 'three' in subcommands


def test_collect_subcommands_fails_with_duplicate_subcommand():
    parser = argparse.ArgumentParser(prog='tests')
    subparser = parser.add_subparsers(dest='command')
    with pytest.raises(SystemExit) as exc:
        subcommands = collect_commands(
            subparser,
            mock_entry_points_iter(['test', 'test']),
        )
    assert exc.type == SystemExit
