#  Copyright 2019 Francesco Ceccon
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.

# pylint: skip-file
import pathlib

import pytest

from galini.branch_and_cut.solver import BranchAndBoundSolver
from galini.galini import Galini
from galini.math import is_close
from galini.pyomo import read_pyomo_model, problem_from_pyomo_model
from galini.timelimit import start_timelimit, set_timelimit


@pytest.mark.parametrize('model_name', [
    'nvs03',
])
def test_ipopt_solver(model_name):
    current_dir = pathlib.Path(__file__).parent
    osil_file = current_dir / 'models' / (model_name + '.osil')
    pyomo_model = read_pyomo_model(osil_file)
    problem = problem_from_pyomo_model(pyomo_model)

    atol = rtol = 1e-4

    galini = Galini()
    galini.update_configuration({
        'galini': {
            'constraint_violation_tol': 1e-2,
        },
        'logging': {
            'stdout': True,
        },
        'branch_and_cut': {
            'tolerance': atol,
            'relative_tolerance': rtol,
            'root_node_feasible_solution_search_timelimit': 0,
            'cuts': {
                'maxiter': 100,
            }
        },
        'cuts_generator': {
            'generators': ['outer_approximation'],
        },
        'ipopt': {
            'ipopt': {
                'acceptable_constr_viol_tol': 1e-3
            },
        },
    })
    set_timelimit(30)
    start_timelimit()
    solver = BranchAndBoundSolver(galini)
    solver.before_solve(pyomo_model, problem)
    solution = solver.solve(problem)

    assert solution.status.is_success()

    sol_file = current_dir / 'solutions' / (model_name + '.sol')
    expected_solution = read_solution(sol_file)

    expected_objective = expected_solution['objective']
    assert solution.objective is not None
    assert is_close(
        expected_objective, solution.objective.value, atol=atol, rtol=rtol
    )

    expected_variables = expected_solution['variables']

    for var_sol in solution.variables:
        assert is_close(
            expected_variables[var_sol.name], var_sol.value,
            atol=atol, rtol=rtol,
        )


def read_solution(filename):
    variables = dict()
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
                variables[var_name] = var_value

    return {
        'objective': objective,
        'variables': variables,
    }
