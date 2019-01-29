# pylint: skip-file
import pyomo.environ as aml
import math


def get_pyomo_model():
    m = aml.ConcreteModel()

    m.x1 = aml.Var()
    m.x2 = aml.Var(bounds=(-1.0, 1.0))
    m.x3 = aml.Var(bounds=(1.0, 2.0))
    m.obj = aml.Objective(expr=m.x1**2 + (m.x2+m.x3)**4 + m.x1*m.x3+m.x2*aml.sin(m.x1+m.x3)+m.x2)
    m.c0 = aml.Constraint(expr=m.x2 >= -1)
    return m
