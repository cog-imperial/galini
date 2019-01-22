# pylint: skip-file
import pytest
import numpy as np
import pyomo.environ as aml
from galini.pyomo import dag_from_pyomo_model
from suspect.interval import Interval as I
from galini.core import (
    Problem,
    Variable,
    Constant,
    Domain,
    LinearExpression,
    LogExpression,
    SqrtExpression,
    SumExpression,
    ProductExpression,
    NegationExpression,
    JacobianEvaluator,
    ForwardJacobianEvaluator,
    ReverseJacobianEvaluator,
    HessianEvaluator,
)

def all_interval_close(expected, actual, tol=1e-5):
    errs = np.array([abs(exp - act).size() for exp, act in zip(expected, actual)])
    assert np.all(errs < tol)


class TestExpressionTreeData(object):
    def _problem(self):
        model = aml.ConcreteModel()
        model.I = range(2)
        model.x = aml.Var(model.I)
        model.obj = aml.Objective(expr=sum(3.0 * model.x[i] for i in model.I))
        model.c0 = aml.Constraint(expr=sum(model.x[i] * model.x[i] for i in model.I) >= 0)
        return dag_from_pyomo_model(model)

    def test_expression_tree_data_of_problem(self):
        dag = self._problem()
        tree_data = dag.expression_tree_data()
        assert len(dag.vertices) == len(tree_data.vertices())

    def test_expression_tree_of_expresssion(self):
        dag = self._problem()
        cons = dag.constraint('c0')
        tree_data = cons.root_expr.expression_tree_data()
        assert len(tree_data.vertices()) == 2 + 2 + 1

        f = tree_data.eval([10.0, 20.0])

        f_x = f.forward(0, [10.0, 20.0])
        assert np.allclose(f_x, [10.0**2 + 20.0**2])

        df_dx = f.reverse(1, [1.0])
        assert np.allclose(df_dx, [2*10.0, 2*20.0])

        H = f.hessian([10.0, 20.0], [1.0])
        assert np.allclose(H, [2, 0, 0, 2])

    def test_expression_tree_of_standalone_expression(self):
        x = Variable('x', None, None, None)
        y = Variable('y', None, None, None)
        w = LinearExpression([x, y], [1.0, 2.0], 0.0)
        z = LogExpression([w])
        tree_data = z.expression_tree_data()
        assert len(tree_data.vertices()) == 1 + 1 + 2

        f = tree_data.eval([4.0, 5.0])
        g = tree_data.eval([I(4.0, 4.0), I(5.0, 5.0)])

        f_x = f.forward(0, [4.0, 5.0])
        expected_f_x = [np.log(4.0 + 2*5.0)]
        assert np.allclose(f_x, expected_f_x)
        g_x = g.forward(0, [I(4.0, 4.0), I(5.0, 5.0)])
        all_interval_close(expected_f_x, g_x)

        df_dx = f.reverse(1, [1.0])
        expected_df_dx = [1/(4.0 + 2*5.0), 2/(4.0 + 2*5.0)]
        assert np.allclose(df_dx, expected_df_dx)
        dg_dx = g.reverse(1, [1.0])
        all_interval_close(expected_df_dx, dg_dx)

        H = f.hessian([4.0, 5.0], [1.0])
        expected_H = [-1/(4.0 + 2*5.0)**2, -2/(4.0 + 2*5.0)**2,
                      -2/(4.0 + 2*5.0)**2, -4/(4.0 + 2*5.0)**2]
        assert np.allclose(H, expected_H)
        H_g = g.hessian([I(4.0, 4.0), I(5.0, 5.0)], [1.0])
        all_interval_close(expected_H, H_g)

    def test_tls2_regression(self):
        """tls2 from MINLPLib2.

        sqrt(i*x) is problematic because x and i can be 0.
        """
        i = Variable('i', None, None, None)
        j = Variable('i', None, None, None)
        x = Variable('x', None, None, None)
        y = Variable('y', None, None, None)

        prod_ix = ProductExpression([i, x])
        sqrt_prod_ix = SqrtExpression([prod_ix])
        prod_jy = ProductExpression([j, y])
        sqrt_prod_jy = SqrtExpression([prod_jy])
        sum_ = SumExpression([sqrt_prod_ix, sqrt_prod_jy])
        negation = NegationExpression([sum_])

        tree_data = negation.expression_tree_data()
        f = tree_data.eval(np.ones(4) * 1e-5)
        d_f = f.reverse(1, [1.0])
        assert np.all(np.isfinite(d_f))
