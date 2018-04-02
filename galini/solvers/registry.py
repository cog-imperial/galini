"""Solvers registry module."""
from galini.registry import Registry


class SolversRegistry(Registry):
    """Registry of MINLP Solvers."""
    def group_name(self):
        return 'galini.solvers'
