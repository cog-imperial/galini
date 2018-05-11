import pytest
import pathlib
import io
import pyomo.environ as aml
from pypopt import PythonJournal, EJournalLevel
from galini import GaliniConfig
from galini.pyomo import read_pyomo_model, dag_from_pyomo_model
from galini.nlp import IpoptNLPSolver


def derivative_check(model):
    config = GaliniConfig()
    solver = IpoptNLPSolver(config, None, None)
    # activate derivative checker
    options = solver.app.options()
    options.set_string_value('derivative_test', 'second-order')
    # setup Ipopt journalist
    output_str = io.StringIO()
    journalist = solver.app.journalist()
    journalist.delete_all_journals()
    journalist.add_journal(PythonJournal(EJournalLevel.J_WARNING, output_str))

    solver.solve(model)
    output = output_str.getvalue()
    output_str.close()

    check_ok = 'No errors detected by derivative checker.' in output
    if not check_ok:
        print(output)
        assert check_ok



@pytest.mark.parametrize('model_name', [
    'hs071.py'
])
def test_osil_model(model_name):
    current_dir = pathlib.Path(__file__).parent
    osil_file = current_dir / 'models' / model_name

    pyomo_model = read_pyomo_model(osil_file)
    dag = dag_from_pyomo_model(pyomo_model)
    derivative_check(dag)
