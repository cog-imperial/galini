# pylint: skip-file
import pyomo.environ as aml
import math


def get_pyomo_model():
    m = aml.ConcreteModel()

    m.x = aml.Var(bounds=(0, None), initialize=10.0)
    m.y = aml.Var(bounds=(0, None), initialize=0.001)

    # wrap in abs to force sum expression
    m.c = aml.Constraint(expr=abs(m.x)+abs(m.y) >= 0)
    m.o = aml.Objective(expr=abs(m.x)+abs(m.y))
    return m
