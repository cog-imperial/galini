# pylint: skip-file
import pytest
import pathlib
import io
import pyomo.environ as aml
from galini import GaliniConfig
from galini.pyomo import read_pyomo_model, problem_from_pyomo_model
from galini.ipopt import IpoptNLPSolver
from galini.galini import Galini
from galini.core import (
    IpoptApplication,
    EJournalLevel,
    PythonJournal,
)


def derivative_check(model_name, model, order):
    galini = Galini()
    galini.update_configuration({
        'galini': {
            'constraint_violation_tol': 1e-2,
        },
        'ipopt': {
            'ipopt': {
                'acceptable_constr_viol_tol': 1e-3,
                'derivative_test': order,
                'max_iter': 1,

            },
            'logging': {
                'level': 'J_ITERSUMMARY',
            }
        },
        'logging': {
            'stdout': True,
            'level': 'DEBUG',
        }
    })
    solver = IpoptNLPSolver(galini)
    solver = IpoptNLPSolver(galini)
    # setup Ipopt journalist
    output_str = io.StringIO()
    app = IpoptApplication()
    journalist = app.journalist()
    journalist.delete_all_journals()
    journalist.add_journal(PythonJournal('Default', EJournalLevel.J_ITERSUMMARY, output_str))

    solver.solve(model, ipopt_application=app)
    output = output_str.getvalue()
    output_str.close()

    check_ok = 'No errors detected by derivative checker.' in output
    if not check_ok:
        print(output)
        assert check_ok


@pytest.mark.parametrize('order', ['first-order', 'only-second-order'])
@pytest.mark.parametrize('model_name', [
    # simple derivative tests
    'dev_product.py', 'dev_division.py', 'dev_sum.py', 'dev_power.py',
    'dev_linear.py', 'dev_negation.py', 'dev_abs.py', 'dev_sqrt.py',
    'dev_exp.py', 'dev_log.py', 'dev_sin.py', 'dev_cos.py', 'dev_tan.py',
    'dev_asin.py', 'dev_acos.py', 'dev_atan.py',
    # Problems from CUTE
    'hs071.py', 'hs078.py', 'cresc4.py', 'eg1.py', 'hatfldf.py', 'hong.py',
    'polak1.py',
])
def test_osil_model(order, model_name):
    if model_name in ['hatfldf.py']:
        pytest.skip('Known derivative fail.')
    current_dir = pathlib.Path(__file__).parent
    osil_file = current_dir / 'models' / model_name

    pyomo_model = read_pyomo_model(osil_file)
    dag = problem_from_pyomo_model(pyomo_model)
    derivative_check(model_name, dag, order)
