import pyomo.environ as pe


def get_pyomo_model(*args, **kwargs):
    """ Returns an example Pyomo model. """
    m = pe.ConcreteModel()

    m.x = pe.Var(bounds=(0, None))
    m.y = pe.Var(bounds=(0, None))
    m.z = pe.Var(bounds=(0, None))

    m.obj = pe.Objective(expr=m.x, sense=pe.maximize)

    m.lin = pe.Constraint(expr=m.x + m.y + m.z == 1)
    m.soc = pe.Constraint(expr=m.x**2 + m.y**2 <= m.z**2)
    m.rot = pe.Constraint(expr=m.x**2 <= m.y * m.z)

    return m


if __name__ == '__main__':
    from galini.galini import Galini

    galini = Galini()
    galini.update_configuration({
        'galini': {
            'timelimit': 100,
        },
        'logging': {
            'stdout': True,
        },
    })

    model = get_pyomo_model()
    solution = galini.solve(model)
    print(solution)
