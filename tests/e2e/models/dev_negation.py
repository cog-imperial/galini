# pylint: skip-file
import pyomo.environ as aml
import math


def get_pyomo_model():
    m = aml.ConcreteModel()

    m.x = aml.Var(initialize=10.0)
    m.c = aml.Constraint(expr=-m.x >= 0)
    m.o = aml.Objective(expr=-m.x)
    return m
