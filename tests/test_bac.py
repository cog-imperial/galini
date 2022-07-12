import math

import pyomo.environ as pe
from galini.galini import Galini


def test_nlp1():
    m = pe.ConcreteModel()

    m.x = pe.Var(bounds=(0, None))
    m.y = pe.Var(bounds=(0, None))
    m.z = pe.Var(bounds=(0, None))

    m.obj = pe.Objective(expr=m.x, sense=pe.maximize)

    m.lin = pe.Constraint(expr=m.x + m.y + m.z == 1)
    m.soc = pe.Constraint(expr=m.x**2 + m.y**2 <= m.z**2)
    m.rot = pe.Constraint(expr=m.x**2 <= m.y * m.z)

    galini = Galini()
    galini.update_configuration({
        'branch_and_cut': {
            'bab': {
              'absolute_gap': 1e-6,
              'relative_gap': 1e-6,
            },
            'mip_solver': {
                'name': 'appsi_gurobi',
            },
        },
    })

    solution = galini.solve(m)
    assert solution.status.is_success()
    assert math.isclose(
        solution.objective_value(),
        0.3269928653,
        rel_tol=1e-4,
        abs_tol=1e-4
    )
