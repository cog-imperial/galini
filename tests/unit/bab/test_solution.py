# pylint: skip-file
import pytest
from galini.solvers import OptimalObjective, OptimalVariable
from galini.branch_and_bound.solution import BabSolution, BabStatusSuccess
from galini.solvers.solution import SolutionPool


class TestSolutionPool:
    def test_add_pool_not_full(self):
        p = SolutionPool(n=5)

        p.add(
            BabSolution(
                BabStatusSuccess(),
                OptimalObjective('obj', 0.0),
                [OptimalVariable('x', 0.0), OptimalVariable('y', 0.0)],
                -10.0,
            )
        )

        assert p.head.objective_value() == 0.0
        assert len(p) == 1

        for i in range(3):
            p.add(
                BabSolution(
                    BabStatusSuccess(),
                    OptimalObjective('obj', -1 * i),
                    [OptimalVariable('x', 0.0), OptimalVariable('y', 0.0)],
                    -10.0,
                )
            )

        assert len(p) == 4
        assert p.head.objective_value() == -2

    def test_add_pool_full(self):
        p = SolutionPool(n=3)
        for i in range(3):
            p.add(
                BabSolution(
                    BabStatusSuccess(),
                    OptimalObjective('obj', i * 1.0),
                    [OptimalVariable('x', 0.0), OptimalVariable('y', 0.0)],
                    -10.0,
                )
            )

        assert len(p) == 3
        assert p.head.objective_value() == 0.0
        assert p[2].objective_value() == 2.0

        p.add(
            BabSolution(
                BabStatusSuccess(),
                OptimalObjective('obj', 1.0),
                [OptimalVariable('x', 0.0), OptimalVariable('y', 0.0)],
                -10.0,
            )
        )

        assert len(p) == 3
        assert p.head.objective_value() == 0.0
        assert p[2].objective_value() == 1.0
