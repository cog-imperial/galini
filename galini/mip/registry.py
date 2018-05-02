"""MIP Solvers registry module."""
from galini.registry import Registry


class MIPSolverRegistry(Registry):
    """Registry of MIP Solvers."""
    def group_name(self):
        return 'galini.mip_solvers'
