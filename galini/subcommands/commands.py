"""CLI Subcommands base class."""
import abc


class CliCommand(metaclass=abc.ABCMeta):
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
