# pylint: skip-file
import pytest
import pathlib
import numpy as np
from galini.solvers import SolversRegistry
from galini.config import ConfigurationManager
from galini.pyomo import read_pyomo_model, problem_from_pyomo_model
from galini.galini import Galini
from galini.ipopt import IpoptNLPSolver


@pytest.mark.parametrize('model_name', [
    'sample', 'gtm', 'harker', 'pollut', 'srcpm',
])
def test_ipopt_solver(model_name):
    current_dir = pathlib.Path(__file__).parent
    osil_file = current_dir / 'models' / (model_name + '.osil')
    pyomo_model = read_pyomo_model(osil_file)
    problem = problem_from_pyomo_model(pyomo_model)

    galini = Galini()
    galini.update_configuration({
        'galini': {
            'constraint_violation_tol': 1e-2,
        },
        'ipopt': {
            'ipopt': {
                'acceptable_constr_viol_tol': 1e-3
            },
        },
    })
    solver = IpoptNLPSolver(galini)
    solution = solver.solve(problem)

    assert solution.status.is_success()

    sol_file = current_dir / 'solutions' / (model_name + '.sol')
    expected_solution = read_solution(sol_file)

    expected_objective = expected_solution['objective']
    assert solution.objective is not None
    assert np.isclose(expected_objective, solution.objective.value)

    if False:
        expected_variables = expected_solution['variables']
        assert len(expected_variables) == len(solution.variables)
        for variable, expected in zip(solution.variables, expected_variables):
            assert np.isclose(expected, variable.value)


def read_solution(filename):
    variables = []
    objective = None
    with open(filename, 'r') as f:
        for line in f:
            parts = line.split()
            var_name = parts[0]
            var_value = float(parts[1])
            if var_name == 'objvar':
                if objective is not None:
                    raise ValueError('Multiple objvar found')
                objective = var_value
            else:
                variables.append(var_value)

    return {
        'objective': objective,
        'variables': variables,
    }
