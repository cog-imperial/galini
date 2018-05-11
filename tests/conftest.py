import mpmath
from galini.pyomo import set_pyomo4_expression_tree


def pytest_sessionstart(session):
    mpmath.mp.dps = 20  # 20 decimal places precision
    set_pyomo4_expression_tree()
