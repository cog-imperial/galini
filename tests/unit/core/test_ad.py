# pylint: skip-file
import pytest
import numpy as np
import pyomo.environ as aml
from galini.pyomo import dag_from_pyomo_model
from galini.core import (
    Problem,
    Variable,
    Constant,
    Domain,
    JacobianEvaluator,
    ForwardJacobianEvaluator,
    ReverseJacobianEvaluator,
    HessianEvaluator,
)


class TestJacobian(object):
    def test_raises_error_on_wrong_input_size(self):
        m = aml.ConcreteModel()
        m.I = range(10)
        m.x = aml.Var(m.I)
        m.obj = aml.Objective(expr=sum(m.x[i] for i in m.I))
        dag = dag_from_pyomo_model(m)
        jac = JacobianEvaluator(dag)
        with pytest.raises(Exception):
            jac.eval_at_x(np.zeros(20), 1)

    def test_jacobian_reverse_mode(self):
        m = aml.ConcreteModel()
        m.I = range(10)
        m.J = range(5)
        m.x = aml.Var(m.I)
        m.obj = aml.Objective(expr=sum(m.x[i]*m.x[i] for i in m.I))
        m.c = aml.Constraint(
            m.J,
            rule=lambda m, j: m.x[j] * sum(m.x[i] for i in m.I if i <= j) + m.x[9-j] * sum(m.x[i] for i in m.I if i >= 9-j) >= 0
        )

        dag = dag_from_pyomo_model(m)

        jac = JacobianEvaluator(dag)
        assert isinstance(jac, ReverseJacobianEvaluator)
        jac.eval_at_x(np.ones(10).astype(np.float64), 1)
        jacobian = jac.jacobian
        assert jacobian.shape == (len(m.J) + 1, len(m.I))

        expected = np.array([
            # x0   x1   x2   x3   x4   x5   x6   x7   x8   x9
            [2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0],
            [2.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 2.0],
            [1.0, 3.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 3.0, 1.0],
            [1.0, 1.0, 4.0, 0.0, 0.0, 0.0, 0.0, 4.0, 1.0, 1.0],
            [1.0, 1.0, 1.0, 5.0, 0.0, 0.0, 5.0, 1.0, 1.0, 1.0],
            [1.0, 1.0, 1.0, 1.0, 6.0, 6.0, 1.0, 1.0, 1.0, 1.0],
        ])
        assert np.array_equal(jacobian, expected)

    def test_jacobian_forward_mode(self):
        m = aml.ConcreteModel()
        m.I = range(3)
        m.J = range(5)

        m.x = aml.Var(m.I)
        m.obj = aml.Objective(expr=sum(m.x[i] for i in m.I))
        m.c = aml.Constraint(m.J, rule=lambda m, j: sum(m.x[i]**(j+1) for i in m.I) >= 0)
        dag = dag_from_pyomo_model(m)

        jac = JacobianEvaluator(dag)
        assert isinstance(jac, ForwardJacobianEvaluator)
        jac.eval_at_x(np.ones(3).astype(np.float64) * 2.0, 1)
        jacobian = jac.jacobian
        assert jacobian.shape == (len(m.J) + 1, len(m.I))

        assert np.array_equal(jacobian[0, :], np.ones(3))
        assert np.array_equal(jacobian[1, :], np.ones(3))
        assert np.array_equal(jacobian[2, :], np.ones(3) * 4.0)
        assert np.array_equal(jacobian[3, :], np.ones(3) * 12.0)
        assert np.array_equal(jacobian[4, :], np.ones(3) * 32.0)
        assert np.array_equal(jacobian[5, :], np.ones(3) * 80.0)


class TestHessian(object):
    def setup_method(self, _func):
        m = aml.ConcreteModel()
        m.I = range(4)
        m.x = aml.Var(m.I)
        m.obj = aml.Objective(expr=m.x[0]*m.x[3]*sum(m.x[i] for i in range(3)) + m.x[2])
        m.c0 = aml.Constraint(expr=m.x[0] * m.x[1] * m.x[2] * m.x[3] >= 25)
        m.c1 = aml.Constraint(expr=sum(m.x[i]*m.x[i] for i in m.I) == 40)
        self.m = m
        self.dag = dag_from_pyomo_model(self.m)
        self.x = np.array([1.0, 5.0, 5.0, 1.0])

    def test_hessian_size(self):
        hes = HessianEvaluator(self.dag)
        hes.eval_at_x(self.x, True)
        hessian = hes.hessian
        assert hessian.shape == (3, 4, 4)

    def test_jacobian_at_x(self):
        hes = HessianEvaluator(self.dag)
        hes.eval_at_x(self.x)
        jacobian = hes.jacobian

        expected = np.array([
            [12.0,  1.0,  2.0, 11.0],
            [25.0,  5.0,  5.0, 25.0],
            [ 2.0, 10.0, 10.0,  2.0],
        ])

        assert np.array_equal(expected, jacobian)

    def test_hessian_at_x(self):
        hes = HessianEvaluator(self.dag)
        hes.eval_at_x(self.x)
        hessian = hes.hessian

        expected_obj = np.array([
            [ 2.0,  1.0,  1.0, 12.0],
            [ 1.0,  0.0,  0.0,  1.0],
            [ 1.0,  0.0,  0.0,  1.0],
            [12.0,  1.0,  1.0,  0.0],
        ])

        assert np.array_equal(expected_obj, hessian[0, :, :])

        expected_c0 = np.array([
            [ 0.0,  5.0,  5.0, 25.0],
            [ 5.0,  0.0,  1.0,  5.0],
            [ 5.0,  1.0,  0.0,  5.0],
            [25.0,  5.0,  5.0,  0.0],
        ])

        assert np.array_equal(expected_c0, hessian[1, :, :])

        expected_c1 = np.diag(np.ones(4) * 2.0)
        assert np.array_equal(expected_c1, hessian[2, :, :])
