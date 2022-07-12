# pylint: skip-file
import pyomo.environ as aml


def get_pyomo_model(*args, **kwargs):
    m = aml.ConcreteModel()

    m.I = range(4)
    m.x = aml.Var(m.I, bounds=(1, 5))

    m.obj = aml.Objective(expr=m.x[0]*m.x[3]*(m.x[0] + m.x[1] + m.x[2]) + m.x[2])

    m.cons0 = aml.Constraint(expr=m.x[0]*m.x[1]*m.x[2]*m.x[3] >= 25)
    m.cons1 = aml.Constraint(expr=sum(m.x[i]**2 for i in m.I) == 40)
    return m
