# pylint: skip-file
import pyomo.environ as aml
import math


def get_pyomo_model():
    m = aml.ConcreteModel()

    m.x = aml.Var(initialize=10.0)
    m.z = aml.Var(initialize=-10.0)

    m.c0 = aml.Constraint(expr=aml.cos(m.x) >= 0)
    m.c1 = aml.Constraint(expr=aml.cos(m.z) >= 0)
    m.o = aml.Objective(expr=aml.cos(m.x))
    return m
