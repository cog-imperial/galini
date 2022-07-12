# pylint: skip-file
import pytest
import numpy as np
from suspect.interval import Interval as I
import pyomo.environ as aml
from galini.pyomo import dag_from_pyomo_model
from galini.interval_hessian import IntervalHessianEvaluator


class BaseTest:
    def get_problem(self, rule):
        m = aml.ConcreteModel()
        m.x = aml.Var()
        m.y = aml.Var()
        m.obj = aml.Objective(expr=rule(m))
        return dag_from_pyomo_model(m)


@pytest.mark.skip('Not updated to work with new DAG')
class TestProduct(BaseTest):
    def test_product(self):
        problem = self.get_problem(lambda m: m.x*m.y)
        h = IntervalHessianEvaluator(problem)
        h.eval_at_x(np.array([I(-1, 1), I(2, 3)]))
        assert np.all(h.jacobian[0] == np.array([I(2, 3), I(-1, 1)]))

        expected_hess = np.array([
            [I(0, 0), I(1, 1)],
            [I(1, 1), I(0, 0)],
        ])
        assert np.all(h.hessian[0] == expected_hess)


@pytest.mark.skip('Not updated to work with new DAG')
class TestPow(BaseTest):
    @pytest.mark.skip(reason='Need more work.')
    def test_Pow(self):
        problem = self.get_problem(lambda m: m.x**2)
        h = IntervalHessianEvaluator(problem)
        h.eval_at_x(np.array([I(-1, 1), I(2, 3)]))

        print(h.jacobian[0])
        print(h.hessian[0])
        assert np.all(h.jacobian[0] == np.array([I(-2, 2), I(0, 0)]))
        assert np.all(h.hessian[0] == np.array([[I(2, 2), I(0, 0)], [I(0, 0), I(0, 0)]]))


@pytest.mark.skip('Not updated to work with new DAG')
class TestNegation(BaseTest):
    def test_negation(self):
        problem = self.get_problem(lambda m: -m.x)
        h = IntervalHessianEvaluator(problem)
        h.eval_at_x(np.array([I(-1, 1), I(2, 3)]))

        print(h.jacobian[0])
        print(h.hessian[0])
        assert np.all(h.jacobian[0] == np.array([I(-1, -1), I(0, 0)]))
        assert np.all(h.hessian[0] == np.array([[I(0, 0), I(0, 0)], [I(0, 0), I(0, 0)]]))


@pytest.mark.skip('Not updated to work with new DAG')
class TestAbs(BaseTest):
    @pytest.mark.skip(reason='Need inequality support in Interval')
    def test_abs_nonnegative(self):
        problem = self.get_problem(lambda m: abs(m.x))
        h = IntervalHessianEvaluator(problem)
        h.eval_at_x(np.array([I(0, 1), I(2, 3)]))

        print(h.jacobian[0])
        print(h.hessian[0])
        assert np.all(h.jacobian[0] == np.array([I(0, 1), I(0, 0)]))
        assert np.all(h.hessian[0] == np.array([[I(0, 0), I(0, 0)], [I(0, 0), I(0, 0)]]))


@pytest.mark.skip(reason='Not same result as paper. Need investigation')
@pytest.mark.skip('Not updated to work with new DAG')
class TestProblemFromAdjimanDallwigFloudasNeumaier(BaseTest):
    def test_problem(self):
        problem = self.get_problem(lambda m: aml.cos(m.x)*aml.sin(m.y) - m.x/(m.y*m.y + 1))

        h = IntervalHessianEvaluator(problem)
        h.eval_at_x(np.array([I(-1, 2), I(-1, 1)]))
        print(h.jacobian[0])
        print(h.hessian[0])
        assert False
