# pylint: skip-file
import numpy as np
import pytest
import pyomo.environ as aml
import galini.core as core
from galini.pyomo.convert import _ComponentFactory, dag_from_pyomo_model
from galini.pyomo.util import model_variables, model_constraints
from suspect.math import inf


class TestConvertVariable(object):
    def test_continuous_variables(self):
        m = aml.ConcreteModel()
        # 10 continuous variables in [-inf, inf]
        m.x = aml.Var(range(10))

        dag = core.RootProblem('test')
        factory = _ComponentFactory(dag)
        count = 0
        for omo_var in model_variables(m):
            new_var = factory.add_variable(omo_var)
            assert new_var.lower_bound is None
            assert new_var.upper_bound is None
            assert new_var.domain == core.Domain.REAL
            count += 1
        assert count == 10

    def test_integer_variables(self):
        m = aml.ConcreteModel()
        # 5 integer variables in [-10, 5]
        m.y = aml.Var(range(5), bounds=(-10, 5), domain=aml.Integers)

        dag = core.RootProblem('test')
        factory = _ComponentFactory(dag)
        count = 0
        for omo_var in model_variables(m):
            new_var = factory.add_variable(omo_var)
            assert new_var.lower_bound == -10
            assert new_var.upper_bound == 5
            assert new_var.domain == core.Domain.INTEGER
            count += 1
        assert count == 5

    def test_binary_variables(self):
        m = aml.ConcreteModel()
        # 10 binary variables
        m.b = aml.Var(range(10), domain=aml.Binary)

        dag = core.RootProblem('test')
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
        for constraint in dag.constraints:
            assert constraint.lower_bound == 0.0
            assert constraint.upper_bound == None
            root = constraint.root_expr
            assert isinstance(root, core.LinearExpression)
            assert root.num_children == 1
            assert root.constant_term == 2.0
            v = root.nth_children(0)
            assert isinstance(v, core.Variable)
            self._check_depth(root)

    def test_nested_expressions(self):
        m = aml.ConcreteModel()
        m.I = range(10)
        m.x = aml.Var(m.I)
        m.y = aml.Var(m.I)

        m.c = aml.Constraint(m.I, rule=lambda m, i: aml.sin(2*m.x[i] - m.y[i]) / (m.x[i] + 1) <= 100)

        dag = dag_from_pyomo_model(m)

        assert len(dag.constraints) == 10
        for constraint in dag.constraints:
            assert constraint.lower_bound == None
            assert constraint.upper_bound == 100
            root = constraint.root_expr
            assert isinstance(root, core.DivisionExpression)
            num, den = root.children
            assert isinstance(num, core.SinExpression)
            assert num.num_children == 1
            num_inner = num.nth_children(0)
            assert isinstance(num_inner, core.LinearExpression)
            assert np.array_equal(np.array(num_inner.coefficients), np.array([2.0, -1.0]))
            assert isinstance(den, core.LinearExpression)
            assert den.constant_term == 1.0
            self._check_depth(root)

    def test_product(self):
        m = aml.ConcreteModel()
        m.I = range(10)
        m.x = aml.Var(m.I)

        m.c = aml.Constraint(range(9), rule=lambda m, i: 2*m.x[i] * 3*m.x[i+1] >= 0)

        dag = dag_from_pyomo_model(m)

        assert len(dag.constraints) == 9
        for constraint in dag.constraints:
            root = constraint.root_expr
            assert isinstance(root, core.ProductExpression)
            assert root.num_children == 2
            linear, var = root.children
            assert isinstance(var, core.Variable)
            assert isinstance(linear, core.LinearExpression)
            assert np.array_equal(linear.coefficients, [6.0])
            self._check_depth(root)

    def test_sum(self):
        m = aml.ConcreteModel()
        m.I = range(10)
        m.x = aml.Var(m.I)

        m.c = aml.Constraint(range(8), rule=lambda m, j: sum(m.x[i]*m.x[i+1] for i in range(j+2)) >= 0)

        dag = dag_from_pyomo_model(m)

        assert len(dag.constraints) == 8
        for constraint in dag.constraints:
            root = constraint.root_expr
            assert isinstance(root, core.SumExpression)
            for c in root.children:
                assert isinstance(c, core.ProductExpression)
            self._check_depth(root)

    def test_negation(self):
        m = aml.ConcreteModel()
        m.I = range(10)
        m.x = aml.Var(m.I)

        m.c = aml.Constraint(expr=-aml.cos(m.x[0]) >= 0)

        dag = dag_from_pyomo_model(m)

        constraint = dag.constraint('c')
        root = constraint.root_expr
        assert isinstance(root, core.NegationExpression)
        self._check_depth(root)

    def test_abs(self):
        m = aml.ConcreteModel()
        m.I = range(10)
        m.x = aml.Var(m.I)

        m.c = aml.Constraint(expr=abs(m.x[0]) >= 0)

        dag = dag_from_pyomo_model(m)

        constraint = dag.constraint('c')
        root = constraint.root_expr
        assert isinstance(root, core.AbsExpression)
        self._check_depth(root)

    def test_pow(self):
        m = aml.ConcreteModel()
        m.I = range(10)
        m.x = aml.Var(m.I)

        m.c0 = aml.Constraint(expr=aml.cos(m.x[0])**2.0 >= 1)
        m.c1 = aml.Constraint(expr=2**aml.sin(m.x[1]) >= 1)

        dag = dag_from_pyomo_model(m)

        c0 = dag.constraint('c0')
        root_c0 = c0.root_expr
        assert isinstance(root_c0, core.PowExpression)
        assert root_c0.num_children == 2
        assert isinstance(root_c0.children[0], core.CosExpression)
        assert isinstance(root_c0.children[1], core.Constant)
        assert root_c0.children[1].value == 2.0

        c1 = dag.constraint('c1')
        root_c1 = c1.root_expr
        assert isinstance(root_c1, core.PowExpression)
        assert root_c1.num_children == 2
        assert isinstance(root_c1.children[0], core.Constant)
        assert isinstance(root_c1.children[1], core.SinExpression)
        self._check_depth(root_c0)
        self._check_depth(root_c1)

    def _check_depth(self, expr):
        nodes = [expr]
        while len(nodes) > 0:
            current = nodes.pop()
            for child in current.children:
                nodes.append(child)
                assert child.depth < current.depth

class TestConvertObjective(object):
    def test_min(self):
        m = aml.ConcreteModel()
        m.I = range(10)
        m.x = aml.Var(m.I)
        m.obj = aml.Objective(expr=sum(m.x[i] for i in m.I))

        dag = dag_from_pyomo_model(m)
        assert len(dag.objectives) == 1
        obj = dag.objective('obj')
        assert isinstance(obj.root_expr, core.LinearExpression)
        assert obj.sense == core.Sense.MINIMIZE

    def test_max(self):
        m = aml.ConcreteModel()
        m.I = range(10)
        m.x = aml.Var(m.I)
        m.obj = aml.Objective(expr=sum(m.x[i] for i in m.I), sense=aml.maximize)

        dag = dag_from_pyomo_model(m)
        assert len(dag.objectives) == 1
        obj = dag.objective('obj')
        assert isinstance(obj.root_expr, core.LinearExpression)
        assert obj.sense == core.Sense.MAXIMIZE
