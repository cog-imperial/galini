"""GALINI exceptions module."""


class DomainError(ValueError):
    """Invalid function domain."""
    pass


class InvalidFileExtensionError(Exception):
    """Exception for invalid input file extension."""
    pass


class InvalidPythonInputError(Exception):
    """Exception for invalid python input file."""
    pass
