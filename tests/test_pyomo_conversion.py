# pylint: skip-file
import numpy as np
import pytest
import pyomo.environ as aml
import galini.core as core
import galini.dag.expressions as dex
from galini.pyomo.convert import _ComponentFactory, dag_from_pyomo_model
from galini.pyomo.util import model_variables, model_constraints
from galini.math.arbitrary_precision import inf


class TestConvertVariable(object):
    def test_continuous_variables(self):
        m = aml.ConcreteModel()
        # 10 continuous variables in [-inf, inf]
        m.x = aml.Var(range(10))

        dag = core.Problem('test')
        factory = _ComponentFactory(dag)
        count = 0
        for omo_var in model_variables(m):
            new_var = factory.add_variable(omo_var)
            assert new_var.lower_bound is None
            assert new_var.upper_bound is None
            assert new_var.domain == core.Domain.REALS
            count += 1
        assert count == 10

    def test_integer_variables(self):
        m = aml.ConcreteModel()
        # 5 integer variables in [-10, 5]
        m.y = aml.Var(range(5), bounds=(-10, 5), domain=aml.Integers)

        dag = core.Problem('test')
        factory = _ComponentFactory(dag)
        count = 0
        for omo_var in model_variables(m):
            new_var = factory.add_variable(omo_var)
            assert new_var.lower_bound == -10
            assert new_var.upper_bound == 5
            assert new_var.domain == core.Domain.INTEGERS
            count += 1
        assert count == 5

    def test_binary_variables(self):
        m = aml.ConcreteModel()
        # 10 binary variables
        m.b = aml.Var(range(10), domain=aml.Binary)

        dag = core.Problem('test')
        factory = _ComponentFactory(dag)
        count = 0
        for omo_var in model_variables(m):
            new_var = factory.add_variable(omo_var)
            assert new_var.lower_bound == 0
            assert new_var.upper_bound == 1
            assert new_var.domain == core.Domain.BINARY
            count += 1
        assert count == 10


class TestConvertExpression(object):
    def test_simple_model(self):
        m = aml.ConcreteModel()
        m.I = range(10)
        m.x = aml.Var(m.I)

        m.c = aml.Constraint(m.I, rule=lambda m, i: m.x[i] + 2 >= 0)

        dag = dag_from_pyomo_model(m)

        assert len(dag.constraints) == 10
        for constraint in dag.constraints.values():
            assert constraint.lower_bound == 0.0
            assert constraint.upper_bound == inf
            root = constraint.root_expr
            assert isinstance(root, core.LinearExpression)
            assert root.num_children == 1
            assert root.constant == 2.0
            v = dag.first_child(root)
            assert isinstance(v, core.Variable)

    def test_nested_expressions(self):
        m = aml.ConcreteModel()
        m.I = range(10)
        m.x = aml.Var(m.I)
        m.y = aml.Var(m.I)

        m.c = aml.Constraint(m.I, rule=lambda m, i: aml.sin(2*m.x[i] - m.y[i]) / (m.x[i] + 1) <= 100)

        dag = dag_from_pyomo_model(m)

        assert len(dag.constraints) == 10
        for constraint in dag.constraints.values():
            assert constraint.lower_bound == -inf
            assert constraint.upper_bound == 100
            root = constraint.root_expr
            assert isinstance(root, core.DivisionExpression)
            num = dag.first_child(root)
            den = dag.second_child(root)
            assert isinstance(num, core.SinExpression)
            assert num.num_children == 1
            num_inner = dag.first_child(num)
            assert isinstance(num_inner, core.LinearExpression)
            assert np.array_equal(np.array(num_inner.coefficients), np.array([2.0, -1.0]))
            assert isinstance(den, core.LinearExpression)
            assert den.constant == 1.0

    def test_product(self):
        m = aml.ConcreteModel()
        m.I = range(10)
        m.x = aml.Var(m.I)

        m.c = aml.Constraint(range(9), rule=lambda m, i: 2*m.x[i] * 3*m.x[i+1] >= 0)

        dag = dag_from_pyomo_model(m)

        assert len(dag.constraints) == 9
        for constraint in dag.constraints.values():
            root = constraint.root_expr
            assert isinstance(root, core.ProductExpression)
            assert root.num_children == 2
            linear = dag.first_child(root)
            var = dag.second_child(root)
            assert isinstance(var, core.Variable)
            assert isinstance(linear, core.LinearExpression)
            assert np.array_equal(linear.coefficients, [6.0])

    def test_sum(self):
        m = aml.ConcreteModel()
        m.I = range(10)
        m.x = aml.Var(m.I)

        m.c = aml.Constraint(range(8), rule=lambda m, j: sum(m.x[i]*m.x[i+1] for i in range(j+2)) >= 0)

        dag = dag_from_pyomo_model(m)

        assert len(dag.constraints) == 8
        for constraint in dag.constraints.values():
            root = constraint.root_expr
            assert isinstance(root, core.SumExpression)
            for c in dag.children(root):
                assert isinstance(c, core.ProductExpression)

    def test_negation(self):
        m = aml.ConcreteModel()
        m.I = range(10)
        m.x = aml.Var(m.I)

        m.c = aml.Constraint(expr=-aml.cos(m.x[0]) >= 0)

        dag = dag_from_pyomo_model(m)

        constraint = dag.constraints['c']
        root = constraint.root_expr
        assert isinstance(root, core.NegationExpression)

    def test_abs(self):
        m = aml.ConcreteModel()
        m.I = range(10)
        m.x = aml.Var(m.I)

        m.c = aml.Constraint(expr=abs(m.x[0]) >= 0)

        dag = dag_from_pyomo_model(m)

        constraint = dag.constraints['c']
        root = constraint.root_expr
        assert isinstance(root, core.AbsExpression)

    def test_pow(self):
        m = aml.ConcreteModel()
        m.I = range(10)
        m.x = aml.Var(m.I)

        m.c0 = aml.Constraint(expr=aml.cos(m.x[0])**2.0 >= 1)
        m.c1 = aml.Constraint(expr=2**aml.sin(m.x[1]) >= 1)

        dag = dag_from_pyomo_model(m)

        c0 = dag.constraints['c0']
        root_c0 = c0.root_expr
        assert isinstance(root_c0, core.PowExpression)
        assert root_c0.num_children == 2
        assert isinstance(dag.first_child(root_c0), core.CosExpression)
        assert isinstance(dag.second_child(root_c0), core.Constant)
        assert dag.second_child(root_c0).value == 2.0

        c1 = dag.constraints['c1']
        root_c1 = c1.root_expr
        assert isinstance(root_c1, core.PowExpression)
        assert root_c1.num_children == 2
        assert isinstance(dag.first_child(root_c1), core.Constant)
        assert isinstance(dag.second_child(root_c1), core.SinExpression)


class TestConvertObjective(object):
    def test_min(self):
        m = aml.ConcreteModel()
        m.I = range(10)
        m.x = aml.Var(m.I)
        m.obj = aml.Objective(expr=sum(m.x[i] for i in m.I))

        dag = dag_from_pyomo_model(m)
        assert len(dag.objectives) == 1
        obj = dag.objectives['obj']
        assert isinstance(obj.root_expr, core.LinearExpression)
        assert obj.sense == core.Sense.MINIMIZE

    def test_max(self):
        m = aml.ConcreteModel()
        m.I = range(10)
        m.x = aml.Var(m.I)
        m.obj = aml.Objective(expr=sum(m.x[i] for i in m.I), sense=aml.maximize)

        dag = dag_from_pyomo_model(m)
        assert len(dag.objectives) == 1
        obj = dag.objectives['obj']
        assert isinstance(obj.root_expr, core.LinearExpression)
        assert obj.sense == core.Sense.MAXIMIZE
