# pylint: skip-file
import pyomo.environ as aml
import math


def get_pyomo_model():
    m = aml.ConcreteModel()
    N = 4

    m.I = range(4)

    m.t = aml.Var(m.I, bounds=(0.5, 1.0))
    m.cons0 = aml.Constraint(expr=sum(m.t[i] for i in m.I) == 1.0)
    m.obj = aml.Objective(expr=(
        0.92+0.08*aml.exp(0.38*25*m.t[0]) -
        2.95+3.95*aml.exp(0.11*50*m.t[1]) -
        1.66+1657834*aml.exp(-1.48*(9.0+4.0*m.t[2])) +
        0.11+0.80*aml.exp(0.00035*20000*m.t[3])
    ))
    return m
