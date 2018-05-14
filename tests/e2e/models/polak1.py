# pylint: skip-file
import pyomo.environ as aml
import math
import operator as op
from functools import reduce


def get_pyomo_model():
    m = aml.ConcreteModel()


    m.I = range(2)


    def init_x(m, i):
        return [50.0, 0.05][i]
    m.x = aml.Var(m.I, initialize=init_x)
    m.u = aml.Var()

    m.cons0 = aml.Constraint(expr=aml.exp(0.001*m.x[0]**2 + m.x[1]**2 - 1.0) <= m.u)
    m.cons1 = aml.Constraint(expr=aml.exp(0.001*m.x[0]**2 + m.x[1]**2 + 1.0) <= m.u)

    m.obj = aml.Objective(expr=m.u)

    return m
