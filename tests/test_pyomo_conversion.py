import pytest
import pyomo.environ as aml
from galini.dag import ProblemDag
import galini.dag.expressions as dex
from galini.pyomo.convert import ComponentFactory, dag_from_pyomo_model
from galini.pyomo.util import model_variables, model_constraints
from galini.math.arbitrary_precision import inf


class TestConvertVariable(object):
    def test_continuous_variables(self):
        m = aml.ConcreteModel()
        # 10 continuous variables in [-inf, inf]
        m.x = aml.Var(range(10))

        dag = ProblemDag()
        factory = ComponentFactory(dag)
        count = 0
        for omo_var in model_variables(m):
            new_var = factory.variable(omo_var)
            assert new_var.name.startswith('x')
            assert new_var.lower_bound is None
            assert new_var.upper_bound is None
            assert new_var.domain == dex.Domain.REALS
            count += 1
        assert count == 10

    def test_integer_variables(self):
        m = aml.ConcreteModel()
        # 5 integer variables in [-10, 5]
        m.y = aml.Var(range(5), bounds=(-10, 5), domain=aml.Integers)

        dag = ProblemDag()
        factory = ComponentFactory(dag)
        count = 0
        for omo_var in model_variables(m):
            new_var = factory.variable(omo_var)
            assert new_var.name.startswith('y')
            assert new_var.lower_bound == -10
            assert new_var.upper_bound == 5
            assert new_var.domain == dex.Domain.INTEGERS
            count += 1
        assert count == 5

    def test_binary_variables(self):
        m = aml.ConcreteModel()
        # 10 binary variables
        m.b = aml.Var(range(10), domain=aml.Binary)

        dag = ProblemDag()
        factory = ComponentFactory(dag)
        count = 0
        for omo_var in model_variables(m):
            new_var = factory.variable(omo_var)
            assert new_var.name.startswith('b')
            assert new_var.lower_bound == 0
            assert new_var.upper_bound == 1
            assert new_var.domain == dex.Domain.BINARY
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
            assert len(constraint.children) == 1
            root = constraint.children[0]
            assert isinstance(root, dex.LinearExpression)
            assert len(root.children) == 1
            assert root.constant_term == 2.0
            assert isinstance(root.children[0], dex.Variable)

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
            assert len(constraint.children) == 1
            root = constraint.children[0]
            assert isinstance(root, dex.DivisionExpression)
            num, den = root.children
            assert isinstance(num, dex.SinExpression)
            assert len(num.children) == 1
            num_inner = num.children[0]
            assert isinstance(num_inner, dex.LinearExpression)
            assert num_inner.coefficients == [2.0, -1.0]
            assert isinstance(den, dex.LinearExpression)
            assert den.constant_term == 1.0

    def test_product(self):
        m = aml.ConcreteModel()
        m.I = range(10)
        m.x = aml.Var(m.I)

        m.c = aml.Constraint(range(9), rule=lambda m, i: 2*m.x[i] * 3*m.x[i+1] >= 0)

        dag = dag_from_pyomo_model(m)

        assert len(dag.constraints) == 9
        for constraint in dag.constraints.values():
            root = constraint.children[0]
            assert isinstance(root, dex.ProductExpression)
            assert len(root.children) == 2
            linear = root.children[0]
            var = root.children[1]
            assert isinstance(var, dex.Variable)
            assert isinstance(linear, dex.LinearExpression)
            assert linear.coefficients == [6.0]

    def test_sum(self):
        m = aml.ConcreteModel()
        m.I = range(10)
        m.x = aml.Var(m.I)

        m.c = aml.Constraint(range(8), rule=lambda m, j: sum(m.x[i]*m.x[i+1] for i in range(j+2)) >= 0)

        dag = dag_from_pyomo_model(m)

        assert len(dag.constraints) == 8
        for constraint in dag.constraints.values():
            root = constraint.children[0]
            assert isinstance(root, dex.SumExpression)
            for c in root.children:
                assert isinstance(c, dex.ProductExpression)

    def test_negation(self):
        m = aml.ConcreteModel()
        m.I = range(10)
        m.x = aml.Var(m.I)

        m.c = aml.Constraint(expr=-aml.cos(m.x[0]) >= 0)

        dag = dag_from_pyomo_model(m)

        constraint = dag.constraints['c']
        root = constraint.children[0]
        assert isinstance(root, dex.NegationExpression)

    def test_abs(self):
        m = aml.ConcreteModel()
        m.I = range(10)
        m.x = aml.Var(m.I)

        m.c = aml.Constraint(expr=abs(m.x[0]) >= 0)

        dag = dag_from_pyomo_model(m)

        constraint = dag.constraints['c']
        root = constraint.children[0]
        assert isinstance(root, dex.AbsExpression)

    def test_pow(self):
        m = aml.ConcreteModel()
        m.I = range(10)
        m.x = aml.Var(m.I)

        m.c0 = aml.Constraint(expr=aml.cos(m.x[0])**2.0 >= 1)
        m.c1 = aml.Constraint(expr=2**aml.sin(m.x[1]) >= 1)

        dag = dag_from_pyomo_model(m)

        c0 = dag.constraints['c0']
        root_c0 = c0.children[0]
        assert isinstance(root_c0, dex.PowExpression)
        assert len(root_c0.children) == 2
        assert isinstance(root_c0.children[0], dex.CosExpression)
        assert isinstance(root_c0.children[1], dex.Constant)
        assert root_c0.children[1].value == 2.0

        c1 = dag.constraints['c1']
        root_c1 = c1.children[0]
        assert isinstance(root_c1, dex.PowExpression)
        assert len(root_c1.children) == 2
        assert isinstance(root_c1.children[0], dex.Constant)
        assert isinstance(root_c1.children[1], dex.SinExpression)


class TestConvertObjective(object):
    def test_min(self):
        m = aml.ConcreteModel()
        m.I = range(10)
        m.x = aml.Var(m.I)
        m.obj = aml.Objective(expr=sum(m.x[i] for i in m.I))

        dag = dag_from_pyomo_model(m)
        assert len(dag.objectives) == 1
        obj = dag.objectives['obj']
        assert isinstance(obj.children[0], dex.LinearExpression)
        assert obj.sense == dex.Sense.MINIMIZE

    def test_max(self):
        m = aml.ConcreteModel()
        m.I = range(10)
        m.x = aml.Var(m.I)
        m.obj = aml.Objective(expr=sum(m.x[i] for i in m.I), sense=aml.maximize)

        dag = dag_from_pyomo_model(m)
        assert len(dag.objectives) == 1
        obj = dag.objectives['obj']
        assert isinstance(obj.children[0], dex.LinearExpression)
        assert obj.sense == dex.Sense.MAXIMIZE
