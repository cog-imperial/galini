# pylint: skip-file
import pytest
import numpy as np
import pyomo.environ as aml
from galini.pyomo import problem_from_pyomo_model
from galini.galini import Galini
from galini.timelimit import set_timelimit
from galini.branch_and_cut.algorithm import BranchAndCutAlgorithm
from galini.branch_and_bound.relaxations import LinearRelaxation
from galini.special_structure import propagate_special_structure, perform_fbbt
from galini.core import Constraint
from galini.sdp.cuts_generator import SdpCutsGenerator


class FakeSolver:
    name = 'branch_and_cut'
    config = {
        'obbt_simplex_maxiter': 100,
    }


def _linear_relaxation(problem):
    bounds = perform_fbbt(
        problem,
        maxiter=10,
        timelimit=60,
    )

    bounds, monotonicity, convexity = \
        propagate_special_structure(problem, bounds)
    return LinearRelaxation(problem, bounds, monotonicity, convexity)


@pytest.fixture()
def problem():
    Q = [[28.0, 23.0, 0.0, 0.0, 0.0, 2.0, 0.0, 24.0],
         [23.0, 0.0, -23.0, -44.0, 10.0, 0.0, 7.0, -7.0],
         [0.0, -23.0, 18.0, 41.0, 0.0, -3.0, -5.0, 2.0],
         [0.0, -44.0, 41.0, -5.0, 5.0, -1.0, 16.0, -50.0],
         [0.0, 10.0, 0.0, 5.0, 0.0, -2.0, -4.0, 21.0],
         [2.0, 0.0, -3.0, -1.0, -2.0, 34.0, -9.0, 20.0],
         [0.0, 7.0, -5.0, 16.0, -4.0, -9.0, 0.0, 0.0],
         [24.0, -7.0, 2.0, -50.0, 21.0, 20.0, 0.0, -45.0]]

    C = [-44, -48, 10, 45, 0, 2, 3, 4, 5]

    Qc = [
        [-28, 13, 5],
        [13, 0, 0],
        [0, 0, 0],
    ]

    Qc2 = [
        [-28, 0, 5],
        [0, 0, 3],
        [0, 3, 0],
    ]

    m = aml.ConcreteModel("model_1")
    m.I = range(8)
    m.x = aml.Var(m.I, bounds=(0, 1))
    m.f = aml.Objective(
        expr=sum(-Q[i][j] * m.x[i] * m.x[j] for i in m.I for j in m.I) + sum(-C[i] * m.x[i] for i in m.I))
    m.c = aml.Constraint(expr=sum(Qc[i][j] * m.x[i] * m.x[j] for i in m.I[0:3] for j in m.I[0:3]) >= -10)
    m.c2 = aml.Constraint(expr=sum(Qc2[i][j] * m.x[i] * m.x[j] for i in m.I[0:3] for j in m.I[0:3]) >= -10)

    return problem_from_pyomo_model(m)


@pytest.mark.parametrize('cut_selection_strategy,expected_solution', [
    ('OPT', [-198.5, -191.13909941730438, -185.6485701612439, -185.63352288462852, -185.61964055492984]),
    # ('FEAS', [-198.5, -195.36202866122858, -185.9984527524324, -185.65131273365154, -185.0030833446988]),
    ('RANDOM', [-198.5, -194.8096267391923, -189.7597741255273, -188.60951478551002, -188.51865496400023]),
    ('COMB_ONE_CON', [-198.5, -191.13909941730438, -185.6485701612439, -185.63352288462852, -185.54225429303932]),
    # ('COMB_ALL_CON', [-198.5, -191.13909941730438, -185.79614907206607, -185.78886388174763, -185.00402059519647]),
])
def test_cut_selection_strategy(problem, cut_selection_strategy, expected_solution):
    galini = Galini()
    relaxation = _linear_relaxation(problem)
    run_id = 'test_run_sdp'

    galini.update_configuration({
        'branch_and_cut': {
            'cuts': {
                'use_lp_cut_phase': True,
                'use_milp_cut_phase': True,
            },
        },
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
    config = galini._config
    sdp_cuts_gen = SdpCutsGenerator(galini, config.cuts_generator.sdp)
    algo = BranchAndCutAlgorithm(galini, FakeSolver(), telemetry=None)
    relaxed_problem = relaxation.relax(problem)
    algo._cuts_generators_manager.before_start_at_root(run_id, problem, None)
    nbs_cuts = []
    mip_sols = []
    if cut_selection_strategy == "RANDOM":
        np.random.seed(0)
    for iteration in range(5):
        set_timelimit(60)
        mip_solution = algo._mip_solver.solve(relaxed_problem)
        assert mip_solution.status.is_success()
        mip_sols.append(mip_solution.objectives[0].value)
        # Generate new cuts
        new_cuts = algo._cuts_generators_manager.generate(run_id, problem, None, relaxed_problem, mip_solution, None, None)
        # Add cuts as constraints
        nbs_cuts.append(len(list(new_cuts)))
        for cut in new_cuts:
            new_cons = Constraint(cut.name, cut.expr, cut.lower_bound, cut.upper_bound)
            relaxation._relax_constraint(problem, relaxed_problem, new_cons)
    assert np.allclose(mip_sols, expected_solution)


def test_sdp_cuts_after_branching(problem):
    galini = Galini()
    galini.update_configuration({
        'cuts_generator': {
            'generators': ['sdp'],
            'sdp': {
                'selection_size': 2,
            },
        }
    })
    run_id = 'test_run_sdp'

    relaxation = _linear_relaxation(problem)

    # Test when branched on x0 in [0.5, 1]
    x0 = problem.variable_view(problem.variables[0])
    x0.set_lower_bound(0.5)
    galini.update_configuration({
        'branch_and_cut': {
            'cuts': {
                'use_lp_cut_phase': True,
                'use_milp_cut_phase': True,
            },
        },
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
    config = galini._config
    sdp_cuts_gen = SdpCutsGenerator(galini, config.cuts_generator.sdp)
    algo = BranchAndCutAlgorithm(galini, FakeSolver(), telemetry=None)
    relaxed_problem = relaxation.relax(problem)
    algo._cuts_generators_manager.before_start_at_root(run_id, problem, None)
    mip_sols = []
    mip_solution = None
    for iteration in range(5):
        set_timelimit(60)
        mip_solution = algo._mip_solver.solve(relaxed_problem)
        assert mip_solution.status.is_success()
        mip_sols.append(mip_solution.objectives[0].value)
        # Generate new cuts
        new_cuts = algo._cuts_generators_manager.generate(run_id, problem, None, relaxed_problem, mip_solution, None, None)
        # Add cuts as constraints
        for cut in new_cuts:
            new_cons = Constraint(cut.name, cut.expr, cut.lower_bound, cut.upper_bound)
            relaxation._relax_constraint(problem, relaxed_problem, new_cons)
    assert np.allclose(mip_sols,
           [-187.53571428571428, -178.17645682147835, -175.10310263115286, -175.0895610878696, -175.03389759123812])
    feas_solution = algo._nlp_solver.solve(relaxed_problem)
    assert(feas_solution.objectives[0].value >= mip_sols[-1])
    sdp_cuts_gen.after_end_at_node(run_id, problem, None, mip_solution)
    sdp_cuts_gen.after_end_at_root(run_id, problem, None, mip_solution)
