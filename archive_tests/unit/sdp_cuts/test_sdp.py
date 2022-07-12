# pylint: skip-file
import pytest
import numpy as np
import pyomo.environ as pe
from collections import namedtuple
from coramin.utils.coramin_enums import RelaxationSide
from suspect.pyomo import create_connected_model
from galini.pyomo.util import update_solver_options, instantiate_solver_with_options
from galini.solvers.solution import load_solution_from_model
from galini.galini import Galini
from galini.branch_and_cut.algorithm import BranchAndCutAlgorithm
from galini.relaxations.relax import relax
from galini.sdp.cuts_generator import SdpCutsGenerator
from galini.relaxations.relax import (update_relaxation_data, rebuild_relaxations, relax_inequality)

FakeStorage = namedtuple('FakeStorage', ['relaxation_data'])
FakeNode = namedtuple('FakeNode', ['storage'])


def _linear_relaxation(problem):
    relaxed, data = relax(problem, use_linear_relaxation=True)
    relaxed.name = problem.name + '_relaxed'
    return relaxed, data


Q = [[28.0, 23.0, 0.0, 0.0, 0.0, 2.0, 0.0, 24.0],
     [23.0, 0.0, -23.0, -44.0, 10.0, 0.0, 7.0, -7.0],
     [0.0, -23.0, 18.0, 41.0, 0.0, -3.0, -5.0, 2.0],
     [0.0, -44.0, 41.0, -5.0, 5.0, -1.0, 16.0, -50.0],
     [0.0, 10.0, 0.0, 5.0, 0.0, -2.0, -4.0, 21.0],
     [2.0, 0.0, -3.0, -1.0, -2.0, 34.0, -9.0, 20.0],
     [0.0, 7.0, -5.0, 16.0, -4.0, -9.0, 0.0, 0.0],
     [24.0, -7.0, 2.0, -50.0, 21.0, 20.0, 0.0, -45.0]]

C = [-44, -48, 10, 45, 0, 2, 3, 4, 5]

Qc = [[-28, 13, 5],
      [13, 0, 0],
      [0, 0, 0]]

Qc2 = [[-28, 0, 5],
       [0, 0, 3],
       [0, 3, 0]]


@pytest.fixture()
def problem():
    m = pe.ConcreteModel("model_1")
    m.I = range(8)
    m.x = pe.Var(m.I, bounds=(0, 1))
    m.f = pe.Objective(
        expr=sum(-Q[i][j] * m.x[i] * m.x[j] for i in m.I for j in m.I) + sum(-C[i] * m.x[i] for i in m.I))
    m.c = pe.Constraint(expr=sum(Qc[i][j] * m.x[i] * m.x[j] for i in m.I[0:3] for j in m.I[0:3]) >= -10)
    m.c2 = pe.Constraint(expr=sum(Qc2[i][j] * m.x[i] * m.x[j] for i in m.I[0:3] for j in m.I[0:3]) >= -10)

    cm, _ = create_connected_model(m)
    return cm



@pytest.mark.parametrize('cut_selection_strategy,expected_solution', [
    ('OPT', [-198.5, -191.13909941730438, -185.6485701612439, -185.63352288462852, -185.61964055492984]),
    ('FEAS', [-198.5, -195.36202866122858, -185.9984527524324, -185.65131273365154, -185.0030833446988]),
    ('RANDOM', [-198.5, -194.8096267391923, -189.7597741255273, -188.60951478551002, -188.51865496400023]),
    ('COMB_ONE_CON', [-198.5, -191.13909941730438, -185.6485701612439, -185.63352288462852, -185.54225429303932]),
    ('COMB_ALL_CON', [-198.5, -191.13909941730438, -185.79614907206607, -185.78886388174763, -185.00402059519647]),
])
def test_cut_selection_strategy(problem, cut_selection_strategy, expected_solution):
    galini = Galini()
    relaxed_problem, relax_data = _linear_relaxation(problem)

    galini.update_configuration({
        'cuts_generator': {
            'generators': ['sdp'],
            'sdp': {
                'domain_eps': 1e-3,
                'thres_sdp_viol': -1e-15,
                'min_sdp_cuts_per_round': 0,
                'max_sdp_cuts_per_round': 5e3,
                'dim': 3,
                'big_m': 10e3,
                'thres_min_opt_sel': 0,
                'selection_size': 4,
                'cut_sel_strategy': cut_selection_strategy,
            },
        }
    })

    storage = FakeStorage(relaxation_data=relax_data)
    node = FakeNode(storage)

    galini.timelimit.start_now()

    config = galini._config
    sdp_cuts_gen = SdpCutsGenerator(galini, config.cuts_generator.sdp)
    sdp_cuts_gen.before_start_at_root(problem, relaxed_problem)

    nbs_cuts = []
    mip_sols = []

    if cut_selection_strategy == "RANDOM":
        np.random.seed(0)

    mip_solver = _instantiate_mip_solver()
    _update_solver_options(mip_solver)

    relaxed_problem._cuts = pe.ConstraintList()

    for iteration in range(5):
        mip_res = mip_solver.solve(relaxed_problem)
        mip_solution = load_solution_from_model(mip_res, relaxed_problem)
        assert mip_solution.status.is_success()
        mip_sols.append(mip_solution.objective)
        # Generate new cuts
        new_cuts = sdp_cuts_gen.generate(problem, relaxed_problem, mip_solution, None, node)

        # Add cuts as constraints
        nbs_cuts.append(len(list(new_cuts)))

        for cut in new_cuts:
            relaxed_cut = relax_inequality(relaxed_problem, cut, RelaxationSide.BOTH, storage.relaxation_data)
            relaxed_problem._cuts.add(relaxed_cut)

        update_relaxation_data(relaxed_problem, storage.relaxation_data)
        rebuild_relaxations(relaxed_problem, storage.relaxation_data, use_linear_relaxation=True)

    assert np.allclose(mip_sols, expected_solution)


def test_sdp_cuts_after_branching(problem):
    galini = Galini()

    # Test when branched on x0 in [0.5, 1]
    x0 = problem.x[0]
    x0.setlb(0.5)
    relaxed_problem, relax_data = _linear_relaxation(problem)

    storage = FakeStorage(relaxation_data=relax_data)
    node = FakeNode(storage)

    galini.update_configuration({
        'cuts_generator': {
            'sdp': {
                'domain_eps': 1e-3,
                'thres_sdp_viol': -1e-15,
                'min_sdp_cuts_per_round': 0,
                'max_sdp_cuts_per_round': 5e3,
                'dim': 3,
                'big_m': 10e3,
                'thres_min_opt_sel': 0,
                'selection_size': 4,
                'cut_sel_strategy': "COMB_ONE_CON"
            },
        }
    })

    galini.timelimit.start_now()

    config = galini._config
    sdp_cuts_gen = SdpCutsGenerator(galini, config.cuts_generator.sdp)
    sdp_cuts_gen.before_start_at_root(problem, relaxed_problem)

    mip_sols = []

    relaxed_problem._cuts = pe.ConstraintList()

    mip_solver = _instantiate_mip_solver()
    _update_solver_options(mip_solver)

    for iteration in range(5):
        mip_res = mip_solver.solve(relaxed_problem)
        mip_solution = load_solution_from_model(mip_res, relaxed_problem)
        assert mip_solution.status.is_success()
        mip_sols.append(mip_solution.objective)
        # Generate new cuts
        new_cuts = sdp_cuts_gen.generate(problem, relaxed_problem, mip_solution, None, node)

        # Add cuts as constraints
        for cut in new_cuts:
            relaxed_cut = relax_inequality(relaxed_problem, cut, RelaxationSide.BOTH, storage.relaxation_data)
            relaxed_problem._cuts.add(relaxed_cut)

        update_relaxation_data(relaxed_problem, storage.relaxation_data)
        rebuild_relaxations(relaxed_problem, storage.relaxation_data, use_linear_relaxation=True)

    assert np.allclose(
        mip_sols,
        [-187.53571428571428, -178.17645682147835, -175.10310263115286, -175.0895610878696, -175.03389759123812]
    )


def _update_solver_options(solver):
    update_solver_options(
        solver,
        timelimit=30,
        absolute_gap=1e-6,
        relative_gap=1e-5,
    )


def _instantiate_mip_solver():
    return instantiate_solver_with_options({
        'name': 'cplex',
        'options': {},
        'timelimit_option': 'timelimit',
        'maxiter_option': 'simplex_limits_iterations',
        'relative_gap_option': 'mip_tolerances_mipgap',
        'absolute_gap_option': 'mip_tolerances_absmipgap',
    })