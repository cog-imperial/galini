# pylint: skip-file
import pyomo.environ as aml
import math


def get_pyomo_model():
    m = aml.ConcreteModel()

    m.x = aml.Var(initialize=10.0)
    m.y = aml.Var(range(10))
    coef = [0.2, 1.0, -3.0, 4.0, -.23, 3.45, 38.128, 1.13, 1e-3, 0.0]
    m.c = aml.Constraint(expr=sum(m.y[i]*coef[i] for i in range(10)) + 0.03 >= 0)
    m.o = aml.Objective(expr=m.x)
    return m
