# pylint: skip-file
import pyomo.environ as aml
import math


def pyomo_create_model():
    m = aml.ConcreteModel()

    m.x = aml.Var(initialize=10.0)
    m.y = aml.Var(initialize=0.001)

    m.c = aml.Constraint(expr=m.x/m.y >= 0)
    m.o = aml.Objective(expr=m.x/m.y)
    return m
