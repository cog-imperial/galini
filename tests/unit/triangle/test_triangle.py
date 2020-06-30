import pytest
import numpy as np
import pyomo.environ as aml
from suspect.pyomo import create_connected_model
from coramin.relaxations.iterators import relaxation_data_objects
from coramin.utils.coramin_enums import RelaxationSide
from coramin.relaxations.mccormick import PWMcCormickRelaxation
from coramin.relaxations.univariate import PWXSquaredRelaxation
from galini.galini import Galini
from galini.branch_and_cut.algorithm import BranchAndCutAlgorithm
from galini.triangle.cuts_generator import TriangleCutsGenerator
from galini.solvers.solution import OptimalObjective, OptimalVariable, Solution, Status, load_solution_from_model
from galini.relaxations.relax import relax


class FakeSolver:
    name = 'branch_and_cut'
    config = {
        'obbt_simplex_maxiter': 100,
    }


class FakeStatus(Status):
    def is_success(self):
        return True

    def is_infeasible(self):
        return False

    def is_unbounded(self):
        return FakeSolver

    def description(self):
        return ''

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


@pytest.fixture()
def problem():
    m = aml.ConcreteModel("model_1")
    m.I = range(8)
    m.x = aml.Var(m.I, bounds=(0, 1))
    m.f = aml.Objective(
        expr=sum(-Q[i][j] * m.x[i] * m.x[j] for i in m.I for j in m.I) + sum(-C[i] * m.x[i] for i in m.I))
    m.c = aml.Constraint(expr=sum(Qc[i][j] * m.x[i] * m.x[j] for i in m.I[0:3] for j in m.I[0:3]) >= -10)

    cm, _ = create_connected_model(m)
    return cm


@pytest.fixture
def galini():
    galini_ = Galini()
    galini_.update_configuration({
        'cuts_generator': {
            'generators': ['triangle'],
            'triangle': {
                'domain_eps': 1e-3,
                'thres_triangle_viol': 1e-7,
                'max_tri_cuts_per_round': 10e3,
                'selection_size': 2,
                'min_tri_cuts_per_round': 0,
            },
        }
    })
    return galini_


def test_adjacency_matrix(galini, problem):
    linear_model, _, _ = relax(problem)
    galini.timelimit.start_now()

    triangle_cuts_gen = TriangleCutsGenerator(galini, galini._config.cuts_generator.triangle)
    triangle_cuts_gen.before_start_at_root(problem, linear_model)
    lower_bounds, upper_bounds, domains, aux_vars, var_by_id, edges = \
        triangle_cuts_gen._detect_bilinear_terms(linear_model)

    expected_adj = [
        [1, 1, 1, 0, 0, 1, 0, 1],
        [1, 0, 1, 1, 1, 0, 1, 1],
        [1, 1, 1, 1, 0, 1, 1, 1],
        [0, 1, 1, 1, 1, 1, 1, 1],
        [0, 1, 0, 1, 0, 1, 1, 1],
        [1, 0, 1, 1, 1, 1, 1, 1],
        [0, 1, 1, 1, 1, 1, 0, 0],
        [1, 1, 1, 1, 1, 1, 0, 1]
    ]

    var_to_idx = aml.ComponentMap()

    for i in problem.I:
        x = linear_model.x[i]
        var_to_idx[x] = i

    for (x_id, y_id) in edges:
        x = var_by_id[x_id]
        y = var_by_id[y_id]
        x_idx = var_to_idx[x]
        y_idx = var_to_idx[y]
        assert expected_adj[x_idx][y_idx] == 1
        # Mark edge as visited
        expected_adj[x_idx][y_idx] = 0

    # Check we visited all edges
    assert np.all(np.isclose(expected_adj, 0))


def test_triange_cut_violations(galini, problem):
    linear_model, _, _ = relax(problem)
    galini.timelimit.start_now()
    triangle_cuts_gen = TriangleCutsGenerator(galini, galini._config.cuts_generator.triangle)
    triangle_cuts_gen.before_start_at_root(problem, linear_model)

    _, _, _, aux_vars, _, _ = \
        triangle_cuts_gen._detect_bilinear_terms(linear_model)

    linear_model.x[0].set_value(0.5)
    linear_model.x[1].set_value(0.5)
    linear_model.x[2].set_value(0.5)
    linear_model.x[3].set_value(0.5)
    linear_model.x[4].set_value(0.5)
    linear_model.x[5].set_value(1.0)
    linear_model.x[6].set_value(0.5)
    linear_model.x[7].set_value(0.5)

    aux_vars_sol = {
        (4, 7): 0.5,
        (5, 6): 0.5,
        (3, 7): 0.0,
        (5, 7): 0.5,
        (3, 6): 0.5,
        (0, 7): 0.5,
        (1, 3): 0.0,
        (3, 4): 0.5,
        (1, 2): 0.0,
        (0, 5): 0.5,
        (0, 0): 0.5,
        (0, 1): 0.5,
        (1, 4): 0.5,
        (3, 3): 0.0,
        (7, 7): 0.0,
        (3, 5): 0.5,
        (1, 6): 0.5,
        (5, 5): 1.0,
        (1, 7): 0.0,
        (2, 2): 0.5,
        (2, 3): 0.5,
        (4, 6): 0.0,
        (2, 5): 0.5,
        (4, 5): 0.5,
        (2, 6): 0.0,
        (2, 7): 0.5,
        (0, 2): 0.5,
    }

    for (i, j), v in aux_vars_sol.items():
        x = linear_model.x[i]
        y = linear_model.x[j]
        w = aux_vars[id(x), id(y)]
        w.set_value(v)

    triangle_viol = triangle_cuts_gen._get_triangle_violations()

    def make_clique(i, j, k):
        return [linear_model.x[i], linear_model.x[j], linear_model.x[k]]

    cliques = [
        make_clique(0, 1, 2),  # 0
        make_clique(0, 1, 7),  # 1
        make_clique(0, 2, 5),  # 2
        make_clique(0, 2, 7),  # 3
        make_clique(0, 5, 7),  # 4
        make_clique(1, 2, 3),  # 5
        make_clique(1, 2, 6),  # 6
        make_clique(1, 2, 7),  # 7
        make_clique(1, 3, 4),  # 8
        make_clique(1, 3, 6),  # 9
        make_clique(1, 3, 7),  # 10
        make_clique(1, 4, 6),  # 11
        make_clique(1, 4, 7),  # 12
        make_clique(2, 3, 5),  # 13
        make_clique(2, 3, 6),  # 14
        make_clique(2, 3, 7),  # 15
        make_clique(2, 5, 6),  # 16
        make_clique(2, 5, 7),  # 17
        make_clique(3, 4, 5),  # 18
        make_clique(3, 4, 6),  # 19
        make_clique(3, 4, 7),  # 20
        make_clique(3, 5, 6),  # 21
        make_clique(3, 5, 7),  # 22
        make_clique(4, 5, 6),  # 23
        make_clique(4, 5, 7),  # 24
    ]

    expected_triangle_viol = [
        [[0.5, -0.5, -0.5], -0.5],
        [[0.5, -0.5, -0.5], -0.5],
        [[0.0, 0.0, -0.5], -0.5],
        [[0.0, 0.0, 0.0], -1.0],
        [[0.0, -0.5, 0.0], -0.5],
        [[-1.0, 0.0, 0.0], 0.0],
        [[0.0, -1.0, 0.0], 0.0],
        [[-1.0, 0.0, 0.0], 0.0],
        [[-0.5, -0.5, 0.5], -0.5],
        [[-0.5, -0.5, 0.5], -0.5],
        [[-0.5, -0.5, -0.5], 0.5],
        [[0.5, -0.5, -0.5], -0.5],
        [[-0.5, 0.5, -0.5], -0.5],
        [[0.0, 0.0, -0.5], -0.5],
        [[-0.5, 0.5, -0.5], -0.5],
        [[0.5, -0.5, -0.5], -0.5],
        [[-0.5, 0.0, -0.5], 0.0],
        [[0.0, -0.5, 0.0], -0.5],
        [[0.0, 0.0, -0.5], -0.5],
        [[0.5, -0.5, -0.5], -0.5],
        [[-0.5, 0.5, -0.5], -0.5],
        [[0.0, -0.5, 0.0], -0.5],
        [[-0.5, 0.0, -0.5], 0.0],
        [[-0.5, 0.0, -0.5], 0.0],
        [[0.0, -0.5, 0.0], -0.5],
    ]

    assert len(triangle_viol) == len(expected_triangle_viol) * 4

    for actual_vars, actual_ineq_type, actual_viol in triangle_viol:
        clique_idx = None
        for i, clique in enumerate(cliques):
            if set([id(c) for c in clique]) == set([id(c) for c in actual_vars]):
                clique_idx = i
                break

        assert clique_idx is not None
        expected_res = expected_triangle_viol[clique_idx]
        if actual_ineq_type == 3:
            assert np.isclose(actual_viol, expected_res[1])
        else:
            any_match = False
            for i, viol in enumerate(expected_res[0]):
                if np.isclose(viol, actual_viol):
                    any_match = True
                    expected_res[0][i] = np.inf
                    break

            assert any_match
