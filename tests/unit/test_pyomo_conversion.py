# pylint: skip-file
import numpy as np
import pytest
import pyomo.environ as aml
import galini.core as core
from galini.pyomo.convert import _ComponentFactory, problem_from_pyomo_model
from galini.pyomo.util import model_variables, model_constraints
from suspect.math import inf


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
            assert new_var.domain == core.Domain.REAL
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
            assert new_var.domain == core.Domain.INTEGER
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

        dag = problem_from_pyomo_model(m)

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

        dag = problem_from_pyomo_model(m)

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
            assert np.isclose(2.0, num_inner.coefficient(num_inner.children[0]))
            assert np.isclose(-1.0, num_inner.coefficient(num_inner.children[1]))
            assert isinstance(den, core.LinearExpression)
            assert den.constant_term == 1.0
            self._check_depth(root)

    def test_product(self):
        m = aml.ConcreteModel()
        m.I = range(10)
        m.x = aml.Var(m.I)

        m.c = aml.Constraint(range(9), rule=lambda m, i: aml.sin(m.x[i]) * aml.cos(m.x[i+1]) >= 0)

        dag = problem_from_pyomo_model(m)

        assert len(dag.constraints) == 9
        for constraint in dag.constraints:
            root = constraint.root_expr
            assert isinstance(root, core.ProductExpression)
            assert root.num_children == 2
            child1, child2 = root.children
            assert isinstance(child1, core.SinExpression)
            assert isinstance(child2, core.CosExpression)
            self._check_depth(root)

    def test_product_as_quadratic(self):
        m = aml.ConcreteModel()
        m.I = range(10)
        m.x = aml.Var(m.I)

        m.c = aml.Constraint(range(9), rule=lambda m, i: 2*m.x[i] * 3*m.x[i+1] >= 0)

        dag = problem_from_pyomo_model(m)

        assert len(dag.constraints) == 9
        for constraint in dag.constraints:
            root = constraint.root_expr
            assert isinstance(root, core.QuadraticExpression)
            assert root.num_children == 2
            assert len(root.terms) == 1
            var1, var2 = root.children
            assert isinstance(var1, core.Variable)
            assert isinstance(var2, core.Variable)
            assert np.isclose(root.coefficient(var1, var2), 6.0)
            self._check_depth(root)

    def test_sum_as_quadratic(self):
        m = aml.ConcreteModel()
        m.I = range(10)
        m.x = aml.Var(m.I)

        m.c = aml.Constraint(range(8), rule=lambda m, j: sum(m.x[i]*m.x[i+1] for i in range(j+2)) >= 0)

        dag = problem_from_pyomo_model(m)

        assert len(dag.constraints) == 8
        for constraint in dag.constraints:
            root = constraint.root_expr
            assert isinstance(root, core.QuadraticExpression)
            self._check_depth(root)

    def test_sum_of_quadratic_with_same_variables(self):
        m = aml.ConcreteModel()
        m.I = range(10)
        m.x = aml.Var(m.I)

        m.c = aml.Objective(expr=2*m.x[0]*m.x[1] + 3*m.x[1]*m.x[0])
        dag = problem_from_pyomo_model(m)

        root_expr = dag.objective.root_expr
        assert isinstance(root_expr, core.QuadraticExpression)
        assert len(root_expr.terms) == 1
        term = root_expr.terms[0]
        assert term.var1 == dag.variable('x[0]')
        assert term.var2 == dag.variable('x[1]')
        assert term.coefficient == 5.0

    def test_sum_of_quadratic_with_same_variables2(self):
        m = aml.ConcreteModel()
        m.I = range(10)
        m.x = aml.Var(m.I)

        m.c = aml.Objective(expr=m.x[0]**2 + m.x[1]**2 + m.x[0])
        dag = problem_from_pyomo_model(m)

        root_expr = dag.objective.root_expr
        assert isinstance(root_expr, core.SumExpression)
        expr = [ex for ex in root_expr.children if isinstance(ex, core.QuadraticExpression)][0]
        assert len(expr.terms) == 2
        for term in expr.terms:
            assert term.coefficient == 1.0

    def test_sum_of_linear_and_other(self):
        m = aml.ConcreteModel()
        m.I = range(10)
        m.x = aml.Var(m.I)

        m.c = aml.Objective(expr=m.x[0] + 2.0*m.x[1] + aml.sin(m.x[2]) + m.x[1])

        dag = problem_from_pyomo_model(m)

        root_expr = dag.objective.root_expr
        assert isinstance(root_expr, core.SumExpression)
        assert len(root_expr.children) == 2
        linear_expr = [ex for ex in root_expr.children if isinstance(ex, core.LinearExpression)][0]
        assert np.isclose(1.0, linear_expr.coefficient(dag.variable('x[0]')))
        assert np.isclose(3.0, linear_expr.coefficient(dag.variable('x[1]')))

    def test_sum_of_quadratic_and_other(self):
        m = aml.ConcreteModel()
        m.I = range(10)
        m.x = aml.Var(m.I)

        m.c = aml.Objective(expr=2*m.x[0]*m.x[1] + 3*m.x[0]*m.x[1] + m.x[3]*m.x[4] + m.x[5] + aml.sin(m.x[6]))

        dag = problem_from_pyomo_model(m)

        root_expr = dag.objective.root_expr
        assert isinstance(root_expr, core.SumExpression)
        assert len(root_expr.children) == 3
        assert len([expr for expr in root_expr.children if isinstance(expr, core.QuadraticExpression)]) == 1
        assert len([expr for expr in root_expr.children if isinstance(expr, core.SinExpression)]) == 1

    def test_sum(self):
        m = aml.ConcreteModel()
        m.I = range(10)
        m.x = aml.Var(m.I)

        m.c = aml.Constraint(range(8), rule=lambda m, j: sum(aml.sin(m.x[i+1]) for i in range(j+2)) >= 0)

        dag = problem_from_pyomo_model(m)

        assert len(dag.constraints) == 8
        for constraint in dag.constraints:
            root = constraint.root_expr
            assert isinstance(root, core.SumExpression)
            for c in root.children:
                assert isinstance(c, core.SinExpression)
            self._check_depth(root)

    def test_negation(self):
        m = aml.ConcreteModel()
        m.I = range(10)
        m.x = aml.Var(m.I)

        m.c = aml.Constraint(expr=-aml.cos(m.x[0]) >= 0)

        dag = problem_from_pyomo_model(m)

        constraint = dag.constraint('c')
        root = constraint.root_expr
        assert isinstance(root, core.NegationExpression)
        self._check_depth(root)

    def test_abs(self):
        m = aml.ConcreteModel()
        m.I = range(10)
        m.x = aml.Var(m.I)

        m.c = aml.Constraint(expr=abs(m.x[0]) >= 0)

        dag = problem_from_pyomo_model(m)

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

        dag = problem_from_pyomo_model(m)

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

        dag = problem_from_pyomo_model(m)
        assert dag.num_objectives == 1
        obj = dag.objective
        assert isinstance(obj.root_expr, core.LinearExpression)
        assert obj.original_sense == core.Sense.MINIMIZE

    def test_max(self):
        m = aml.ConcreteModel()
        m.I = range(10)
        m.x = aml.Var(m.I)
        m.obj = aml.Objective(expr=sum(m.x[i] for i in m.I), sense=aml.maximize)

        dag = problem_from_pyomo_model(m)
        assert dag.num_objectives == 1
        obj = dag.objective
        assert isinstance(obj.root_expr, core.LinearExpression)
        assert obj.original_sense == core.Sense.MAXIMIZE
