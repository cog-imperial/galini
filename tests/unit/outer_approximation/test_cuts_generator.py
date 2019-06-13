# pylint: skip-file
import pytest
import numpy as np
import pyomo.environ as pe
from tests.unit.fixtures import FakeStatus, FakeStatusEnum
from galini.pyomo import dag_from_pyomo_model
from galini.solvers.solution import (
    OptimalObjective,
    OptimalVariable,
    Solution,
)
from galini.outer_approximation.cuts_generator import (
    OuterApproximationCutsGenerator
)
from galini.galini import Galini
from galini.util import print_problem, expr_to_str


@pytest.fixture
def problem():
    m = pe.ConcreteModel()

    m.I = range(2)
    m.x = pe.Var(m.I, bounds=(0, 4))
    m.y = pe.Var(m.I, bounds=(0, 4), domain=pe.Binary)

    @m.Objective()
    def f(m):
        return m.y[0] + m.y[1] + m.x[0]**2 + m.x[1]**2

    @m.Constraint()
    def g0(m):
        return (m.x[0] - 2)**2 - m.x[1] <= 0

    @m.Constraint()
    def g1(m):
        return m.x[0] - 2*m.y[0] >= 0

    @m.Constraint()
    def g2(m):
        return m.x[0] - m.x[1] - 3*(1 - m.y[0]) <= 0

    @m.Constraint()
    def g3(m):
        return m.x[0] - (1 - m.y[0]) >= 0

    @m.Constraint()
    def g4(m):
        return m.x[1] - m.y[1] >= 0

    @m.Constraint()
    def g5(m):
        return m.x[0] + m.x[1] >= 3 * m.y[0]

    @m.Constraint()
    def g6(m):
        return sum(m.y[i] for i in m.I) >= 1

    return dag_from_pyomo_model(m)


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
    solution = Solution(
        FakeStatus(FakeStatusEnum.Success),
        [OptimalObjective('f', 0.0)],
        [
            OptimalVariable('x[0]', 0.0),
            OptimalVariable('x[1]', 0.0),
            OptimalVariable('y[0]', 1.0),
            OptimalVariable('y[1]', 1.0),
        ]
    )
    cuts = generator.generate(
        run_id=0,
        problem=None,
        relaxed_problem=problem,
        mip_solution=solution,
        tree=None,
        node=None,
    )
    print_problem(problem)
    assert len(cuts) == 8
    objective_cuts = [c for c in cuts if c.is_objective]
    constraint_cuts = [c for c in cuts if not c.is_objective]

    assert len(objective_cuts) == 1
    _check_objective_cut(objective_cuts[0])

    assert len(constraint_cuts) == 7
    _cut_map = [
        _check_g0,
        _check_g1,
        _check_g2,
        _check_g3,
        _check_g4,
        _check_g5,
        _check_g6,
    ]
    for i, cut in enumerate(constraint_cuts):
        _cut_map[i](cut)


def _check_objective_cut(cut):
    assert cut.is_objective
    assert cut.lower_bound is None
    assert cut.upper_bound is None
    expr = cut.expr
    coefs = [expr.coefficient(v) for v in expr.children]
    assert np.all(np.isclose(coefs, [4.0, 4.0, 1.0, 1.0]))
    assert np.isclose(expr.constant_term, -8.0)


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
    _check_gx(cut, None, 0, [0.0, -1.0, 0.0, 0.0], 0.0)


def _check_g1(cut):
    _check_gx(cut, 0, None, [1.0, 0.0, -2.0, 0.0], 0.0)


def _check_g2(cut):
    _check_gx(cut, None, 0, [1.0, -1.0, 3.0, 0.0], -3.0)


def _check_g3(cut):
    _check_gx(cut, 0, None, [1.0, 0.0, 1.0, 0.0], -1)


def _check_g4(cut):
    _check_gx(cut, 0, None, [0.0, 1.0, 0.0, -1.0], 0.0)


def _check_g5(cut):
    _check_gx(cut, None, 0.0, [-1.0, -1.0, 3.0, 0.0], 0.0)


def _check_g6(cut):
    _check_gx(cut, 1, None, [0.0, 0.0, 1.0, 1.0], 0.0)
