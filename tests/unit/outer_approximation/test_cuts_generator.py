# pylint: skip-file
import numpy as np
import pyomo.environ as pe
import pytest

from galini.branch_and_bound.relaxations import LinearRelaxation, \
    ConvexRelaxation
from galini.galini import Galini
from galini.outer_approximation import (
    OuterApproximationCutsGenerator
)
from galini.pyomo import problem_from_pyomo_model
from galini.relaxations.relaxed_problem import RelaxedProblem
from galini.solvers.solution import (
    OptimalObjective,
    OptimalVariable,
    Solution,
)
from galini.special_structure import propagate_special_structure
from tests.unit.fixtures import FakeStatus, FakeStatusEnum


@pytest.fixture
def problem():
    m = pe.ConcreteModel()

    m.I = range(2)
    m.x = pe.Var(m.I, domain=pe.Integers, bounds=(0, 200))
    m.x[0].setub(4.0)
    m.x[1].setub(7.0)

    @m.Objective()
    def f(m):
        return m.x[0]**2 - 16*m.x[0] + m.x[1]**2 - 4*m.x[1] + 68

    @m.Constraint()
    def g0(m):
        return m.x[0]**2 + m.x[1] >= 0

    @m.Constraint()
    def g1(m):
        return -0.33*m.x[0] - m.x[1] + 4.5 >= 0

    return problem_from_pyomo_model(m)


def test_outer_approximation_cuts(problem):
    galini = Galini()
    galini.update_configuration({
        'logging': {
            'level': 'DEBUG',
            'stdout': False,
        }
    })
    config = galini.get_configuration_group('cuts_generator.outer_approximation')
    generator = OuterApproximationCutsGenerator(galini, config)

    bounds, mono, cvx = propagate_special_structure(problem)

    cvx_relaxation = ConvexRelaxation(problem, bounds, mono, cvx)
    relaxed_problem = RelaxedProblem(cvx_relaxation, problem).relaxed

    linear_relaxation = LinearRelaxation(relaxed_problem, bounds, mono, cvx)
    linear_problem = RelaxedProblem(linear_relaxation, relaxed_problem).relaxed

    solution = Solution(
        FakeStatus(FakeStatusEnum.Success),
        [OptimalObjective('_objvar', 8.00)],
        [
            OptimalVariable('x[0]', 4.0),
            OptimalVariable('x[1]', 2.0),
            OptimalVariable('_aux_0', 12.0),
            OptimalVariable('_aux_1', 0.0),
            OptimalVariable('_objvar', 8.0),
            OptimalVariable('_aux_bilinear_x[0]_x[0]', 12.0),
            OptimalVariable('_aux_bilinear_x[1]_x[1]', 0.0),
        ]
    )

    cuts = generator.generate(
        run_id=0,
        problem=problem,
        relaxed_problem=relaxed_problem,
        linear_problem=linear_problem,
        mip_solution=solution,
        tree=None,
        node=None,
    )
    assert len(cuts) == 2

    objective_cuts = [c for c in cuts if c.is_objective]
    constraint_cuts = [c for c in cuts if not c.is_objective]

    assert len(objective_cuts) == 0

    assert len(constraint_cuts) == 2
    _cut_map = [
        _check_g0,
        _check_g1,
    ]
    for i, cut in enumerate(constraint_cuts):
        _cut_map[i](cut)


def _check_gx(cut, expected_lb, expected_ub, expected_coefs, expected_const):
    assert not cut.is_objective

    if expected_lb is None:
        assert cut.lower_bound is None
    else:
        assert np.isclose(cut.lower_bound, expected_lb, atol=1e-6)

    if expected_ub is None:
        assert cut.upper_bound is None
    else:
        assert np.isclose(cut.upper_bound, expected_ub, atol=1e-6)

    expr = cut.expr
    coefs = [expr.coefficient(v) for v in expr.children]
    assert np.all(np.isclose(coefs, expected_coefs, atol=1e-6))
    assert np.isclose(expr.constant_term, expected_const, atol=1e-6)


def _check_g0(cut):
    _check_gx(cut, None, 0, [8.0, 0.0, -1.0, 0.0, 0.0], -16.0)


def _check_g1(cut):
    _check_gx(cut, None, 0, [0.0, 4.0, 0.0, -1.0, 0.0], -4.0)
