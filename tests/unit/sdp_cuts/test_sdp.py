import pytest
import numpy as np
import pyomo.environ as aml
from galini.pyomo import dag_from_pyomo_model
from galini.solvers import SolversRegistry
from galini.config import ConfigurationManager
from galini.cuts import CutsGeneratorsRegistry
from galini.abb.relaxation import AlphaBBRelaxation
from galini.bab.branch_and_cut import BranchAndCutAlgorithm
from galini.core import Constraint
from galini.sdp_cuts.generator import SdpCutsGenerator


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

    return dag_from_pyomo_model(m)


def test_sdp_cuts(problem):
    solvers_reg = SolversRegistry()
    solver_cls = solvers_reg.get('ipopt')
    cuts_gen_reg = CutsGeneratorsRegistry()
    config_manager = ConfigurationManager()
    config_manager.initialize(solvers_reg, cuts_gen_reg)
    config = config_manager.configuration
    config.update({
        'cuts_generator': {
            'sdp': {
                'selection_size': 2,
            },
        }
    })
    solver_ipopt = solver_cls(config, solvers_reg, cuts_gen_reg)
    solver_mip = solver_ipopt.instantiate_solver("mip")
    relaxation = AlphaBBRelaxation()

    # Test at root node for all cut selection strategies
    cut_sel_strategies = ["OPT", "FEAS", "RANDOM", "COMB_ONE_CON", "COMB_ALL_CON"]
    mip_sols_to_match = [
        [-198.5, -191.13909941730438, -185.6485701612439, -185.63352288462852, -185.61964055492984],
        [-198.5, -195.36202866122858, -185.9984527524324, -185.65131273365154, -185.0030833446988],
        [-198.5, -194.8096267391923, -189.7597741255273, -188.60951478551002, -188.51865496400023],
        [-198.5, -191.13909941730438, -185.6485701612439, -185.63352288462852, -185.54225429303932],
        [-198.5, -191.13909941730438, -185.79614907206607, -185.78886388174763, -185.00402059519647]
    ]
    for idx, cut_sel_strategy in enumerate(cut_sel_strategies[0:5]):
        config.update({
            'cuts_generator': {
                'sdp': {
                    'selection_size': 4,
                    'cut_sel_strategy': cut_sel_strategy
                },
            }
        })
        sdp_cuts_gen = SdpCutsGenerator(config.cuts_generator.sdp)
        algo = BranchAndCutAlgorithm(solver_ipopt, solver_mip, sdp_cuts_gen, config)
        relaxed_problem = relaxation.relax(problem)
        algo._cuts_generators_manager.before_start_at_root(problem)
        nbs_cuts = []
        mip_sols = []
        if cut_sel_strategy == "RANDOM":
            np.random.seed(0)
        for iteration in range(5):
            mip_solution = algo._mip_solver.solve(relaxed_problem, logger=None)
            assert mip_solution.status.is_success()
            mip_sols.append(mip_solution.objectives[0].value)
            # Generate new cuts
            new_cuts = algo._cuts_generators_manager.generate(problem, relaxed_problem, mip_solution, None, None)
            # Add cuts as constraints
            nbs_cuts.append(len(list(new_cuts)))
            for cut in new_cuts:
                new_cons = Constraint(cut.name, cut.expr, cut.lower_bound, cut.upper_bound)
                relaxation._relax_constraint(problem, relaxed_problem, new_cons)
        print(mip_sols)
        assert (np.allclose(mip_sols, mip_sols_to_match[idx]))

    # Test when branched on x0 in [0.5, 1]
    x0 = problem.variable_view(problem.variables[0])
    x0.set_lower_bound(0.5)
    config.update({
        'cuts_generator': {
            'sdp': {
                'selection_size': 4,
                'cut_sel_strategy': "COMB_ONE_CON"
            },
        }
    })
    sdp_cuts_gen = SdpCutsGenerator(config.cuts_generator.sdp)
    algo = BranchAndCutAlgorithm(solver_ipopt, solver_mip, sdp_cuts_gen, config)
    relaxed_problem = relaxation.relax(problem)
    algo._cuts_generators_manager.before_start_at_root(problem)
    mip_sols = []
    mip_solution = None
    for iteration in range(5):
        mip_solution = algo._mip_solver.solve(relaxed_problem, logger=None)
        assert mip_solution.status.is_success()
        mip_sols.append(mip_solution.objectives[0].value)
        # Generate new cuts
        new_cuts = algo._cuts_generators_manager.generate(problem, relaxed_problem, mip_solution, None, None)
        # Add cuts as constraints
        for cut in new_cuts:
            new_cons = Constraint(cut.name, cut.expr, cut.lower_bound, cut.upper_bound)
            relaxation._relax_constraint(problem, relaxed_problem, new_cons)
    assert(np.allclose(mip_sols,
           [-187.53571428571428, -178.17645682147835, -175.10310263115286, -175.0895610878696, -175.03389759123812]))
    feas_solution = algo._nlp_solver.solve(relaxed_problem, logger=None)
    assert(feas_solution.objectives[0].value >= mip_sols[-1])
    sdp_cuts_gen.after_end_at_node(problem, mip_solution)
    sdp_cuts_gen.after_end_at_root(problem, mip_solution)
