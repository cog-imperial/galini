# pylint: skip-file
import pyomo.environ as aml
import math


def get_pyomo_model():
    m = aml.ConcreteModel()

    m.x = aml.Var(initialize=10.0)
    m.y = aml.Var(initialize=2.0)
    m.z = aml.Var(initialize=0.03)

    m.c0 = aml.Constraint(expr=m.x**2.0 >= 0)
    m.c1 = aml.Constraint(expr=m.x**(-5.0) >= 0)
    m.c2 = aml.Constraint(expr=m.x**m.y >= 0)
    m.c3 = aml.Constraint(expr=m.x**m.z >= 0)
    m.c4 = aml.Constraint(expr=3**m.z >= 0)
    m.o = aml.Objective(expr=m.x)
    return m
