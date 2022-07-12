# pylint: skip-file
import pyomo.environ as aml
import math
import operator as op
from functools import reduce


def get_pyomo_model(*args, **kwargs):
    m = aml.ConcreteModel()

    m.I = range(5)

    def init_x(m, i):
        return [-1.717142, 1.595708, 1.827248, -0.7636429, -0.7636435][i]
    m.x = aml.Var(m.I, initialize=init_x)

    m.cons0 = aml.Constraint(expr=sum(m.x[i]**2 for i in m.I) == 10)
    m.cons1 = aml.Constraint(expr=m.x[1]*m.x[2] - 5*m.x[3]*m.x[4] == 0)
    m.cons2 = aml.Constraint(expr=m.x[0]**3 + m.x[1]**3 == -1)
    m.obj = aml.Objective(expr=reduce(op.mul, [m.x[i] for i in m.I], 1.0))

    return m
