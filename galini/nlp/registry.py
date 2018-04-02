"""NLP Solvers registry module."""
from galini.registry import Registry


class NLPSolversRegistry(Registry):
    """Registry of NLP Solvers"""
    def group_name(self):
        return 'galini.nlp_solvers'
