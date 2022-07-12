# pylint: skip-file
import pyomo.environ as aml
import math


def get_pyomo_model():
    m = aml.ConcreteModel()
    m.x1 = aml.Var(bounds=(1, None), initialize=2.0)
    m.x2 = aml.Var(bounds=(1, None), initialize=2.0)
    m.x3 = aml.Var(bounds=(1, None), initialize=2.0)

    m.ov = aml.Var(initialize=0.0)

    m.obj = aml.Objective(expr=m.ov)
    m.e1 = aml.Constraint(expr=aml.exp(m.x3)*m.x2 + m.x1 == 0.032)
    m.e2 = aml.Constraint(expr=aml.exp(2*m.x3)*m.x2 + m.x1 == 0.056)
    m.e3 = aml.Constraint(expr=aml.exp(3*m.x3)*m.x2 + m.x1 == 0.099)

    return m
