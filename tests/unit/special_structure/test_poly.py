# pylint: skip-file
import pytest
import pyomo.environ as aml
from galini.pyomo import dag_from_pyomo_model
from galini.special_structure.poly import detect_polynomial_degree


@pytest.fixture()
def model():
    m = aml.ConcreteModel()
    m.I = range(5)
    m.x = aml.Var(m.I)
    return m


class TestPolynomialDegree:
    def test_linear(self, model):
        model.obj = aml.Objective(expr=sum(model.x[i] for i in model.I))
        self._assert_poly(model, lambda p: p.is_linear())

    def test_division_by_const(self, model):
        model.obj = aml.Objective(expr=model.x[0] / 2.0)
        self._assert_poly(model, lambda p: p.is_linear())

    def test_division_by_noncost(self, model):
        model.obj = aml.Objective(expr=model.x[0] / model.x[1])
        self._assert_poly(model, lambda p: not p.is_polynomial())

    def test_product(self, model):
        model.obj = aml.Objective(expr=model.x[0] * model.x[1] * model.x[2] * model.x[3] * model.x[4])
        self._assert_poly(model, lambda p: p.is_polynomial() and p.degree == 5)

    def test_pow_by_const(self, model):
        model.obj = aml.Objective(expr=model.x[0] ** 3)
        self._assert_poly(model, lambda p: p.is_polynomial() and p.degree == 3)

    @pytest.mark.parametrize('func', [aml.cos, aml.sin, aml.tan, aml.acos, aml.asin, aml.atan])
    def test_unary_function(self, model, func):
        model.obj = aml.Objective(expr=func(model.x[0]))
        self._assert_poly(model, lambda p: not p.is_polynomial())

    def _assert_poly(self, model, func):
        dag = dag_from_pyomo_model(model)
        poly_degree = detect_polynomial_degree(dag)
        obj = dag.objective('obj')
        obj_expr = obj.root_expr
        obj_poly = poly_degree[obj_expr]
        assert func(obj_poly)
