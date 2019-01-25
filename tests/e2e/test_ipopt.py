# pylint: skip-file
import pytest
import pathlib
import numpy as np
from galini import GaliniConfig
from galini.pyomo import read_pyomo_model, dag_from_pyomo_model
from galini.ipopt import IpoptNLPSolver


@pytest.mark.parametrize('model_name', [
    'sample', 'gtm', 'harker', 'pollut', 'srcpm',
])
def test_ipopt_solver(model_name):
    current_dir = pathlib.Path(__file__).parent
    osil_file = current_dir / 'models' / (model_name + '.osil')
    pyomo_model = read_pyomo_model(osil_file)
    problem = dag_from_pyomo_model(pyomo_model)

    config = GaliniConfig()
    solver = IpoptNLPSolver(config, None)
    solution = solver.solve(problem)

    assert solution.status.is_success()

    sol_file = current_dir / 'solutions' / (model_name + '.sol')
    expected_solution = read_solution(sol_file)

    expected_objectives = expected_solution['objectives']
    assert len(expected_objectives) == len(solution.objectives)
    for objective, expected in zip(solution.objectives, expected_objectives):
        assert np.isclose(expected, objective.value)

    if False:
        expected_variables = expected_solution['variables']
        assert len(expected_variables) == len(solution.variables)
        for variable, expected in zip(solution.variables, expected_variables):
            assert np.isclose(expected, variable.value)


def read_solution(filename):
    variables = []
    objectives = []
    with open(filename, 'r') as f:
        for line in f:
            parts = line.split()
            var_name = parts[0]
            var_value = float(parts[1])
            if var_name == 'objvar':
                if len(objectives) > 0:
                    raise ValueError('Multiple objvar found')
                objectives.append(var_value)
            else:
                variables.append(var_value)

    return {
        'objectives': objectives,
        'variables': variables,
    }
