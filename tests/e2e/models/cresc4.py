# pylint: skip-file
import pyomo.environ as aml
import math


def get_pyomo_model():
    m = aml.ConcreteModel()

    N = 4
    NP = 100
    PI_2 = 2.0 * math.atan(1.0)
    X = [1.0, 0.0, 0.0, 0.5]
    Y = [0.0, 1.0, -1.0, 0.0]

    m.I = range(N)

    m.v1 = aml.Var(initialize=-40.0)
    m.w1 = aml.Var(initialize=5.0)
    m.d = aml.Var(bounds=(1e-8, None), initialize=1.0)
    m.a = aml.Var(bounds=(1.0, None), initialize=2.0)
    m.t = aml.Var(bounds=(0.0, 6.2831852), initialize=1.5)
    m.r = aml.Var(bounds=(0.39, None), initialize=0.75)
    m.f = aml.Var()
    m.s1 = aml.Var(bounds=(-0.99, 0.99))
    m.s2 = aml.Var(bounds=(-0.99, 0.99))

    def eq_1(m, i):
        return (
            (m.v1+m.a*m.d*aml.cos(m.t) - X[i])**2 +
            (m.w1+m.a*m.d*aml.sin(m.t) - Y[i])**2 -
            (m.d+m.r)**2 <= 0
        )
    m.eq_1 = aml.Constraint(m.I, rule=eq_1)

    def eq_2(m, i):
        return (m.v1 - X[i])**2 + (m.w1 - Y[i])**2 - (m.a*m.d+m.r)**2 >= 0.0
    m.eq_2 = aml.Constraint(m.I, rule=eq_2)

    m.add_1 = aml.Constraint(expr=(
        m.s1 == -((m.a+m.d)**2 - (m.a*m.d+m.r)**2 + (m.d+m.r)**2)/(2*(m.d+m.r)*m.a*m.d)
    ))

    m.add_2 = aml.Constraint(expr=(
        m.s2 == ((m.a+m.d)**2 + (m.a*m.d+m.r)**2 - (m.d+m.r)**2)/(2*(m.a*m.d+m.r)*m.a*m.d)
    ))

    m.obj = aml.Objective(expr=(
        (m.d+m.r)*(m.d+m.r)*(PI_2-aml.atan(m.s1/(aml.sqrt(1-m.s1*m.s1)))) -
        (m.a*m.d+m.r)*(m.a*m.d+m.r)*(PI_2-aml.atan(m.s2/(aml.sqrt(1-m.s2*m.s2)))) +
        (m.d+m.r)*m.a*m.d*aml.sin((PI_2-aml.atan(m.s1/(aml.sqrt(1-m.s1*m.s1)))))

    ))
    return m
