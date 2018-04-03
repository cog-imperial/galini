from numbers import Number
import pyomo.environ as aml
from galini.pyomo.util import (
    model_variables,
    model_objectives,
    model_constraints,
    bounds_and_expr,
)
from galini.pyomo.expr_visitor import ExpressionHandler, bottom_up_visit
from galini.pyomo.expr_dict import ExpressionDict
from galini.float_hash import BTreeFloatHasher
from galini.dag.dag import ProblemDag
import galini.dag.expressions as dex


def dag_from_pyomo_model(model):
    """Convert the Pyomo ``model`` to GALINI DAG.

    Parameters
    ----------
    model : ConcreteModel
        the Pyomo model.

    Returns
    -------
    ProblemDag
        GALINI problem DAG.
    """
    dag = ProblemDag(name=model.name)
    factory = ComponentFactory(dag)
    for omo_var in model_variables(model):
        new_var = factory.variable(omo_var)
        dag.add_variable(new_var)

    for omo_cons in model_constraints(model):
        new_cons = factory.constraint(omo_cons)
        dag.add_constraint(new_cons)

    for omo_obj in model_objectives(model):
        new_obj = factory.objective(omo_obj)
        dag.add_objective(new_obj)

    return dag


def convert_domain(dom):
    if isinstance(dom, aml.RealSet):
        return dex.Domain.REALS
    elif isinstance(dom, aml.IntegerSet):
        return dex.Domain.INTEGERS
    elif isinstance(dom, aml.BooleanSet):
        return dex.Domain.BINARY


def convert_expression(memo, dag, expr):
    handler = ExpressionConverterHandler(memo, dag)
    bottom_up_visit(handler, expr)
    return memo[expr]


class ExpressionConverterHandler(ExpressionHandler):
    def __init__(self, memo, dag):
        self.memo = memo
        self.dag = dag

    def get(self, expr):
        if isinstance(expr, Number):
            const = aml.NumericConstant(expr)
            return self.get(const)
        else:
            return self.memo[expr]

    def set(self, expr, new_expr):
        self.memo[expr] = new_expr
        if len(new_expr.children) > 0:
            for child in new_expr.children:
                child.add_parent(new_expr)
        self.dag.add_vertex(new_expr)

    def _check_children(self, expr):
        for a in expr._args:
            if self.get(a) is None:
                raise RuntimeError('unknown child')

    def visit_number(self, n):
        const = aml.NumericConstant(n)
        self.visit_numeric_constant(const)

    def visit_numeric_constant(self, expr):
        if self.memo[expr] is not None:
            return self.memo[expr]
        const = dex.Constant(expr.value)
        self.set(expr, const)

    def visit_variable(self, expr):
        if self.memo[expr] is not None:
            return self.memo[expr]
        raise AssertionError('Unknown variable encountered')

    def visit_equality(self, expr):
        raise AssertionError('Invalid EqualityExpression encountered')

    def visit_inequality(self, expr):
        raise AssertionError('Invalid EqualityExpression encountered')

    def visit_product(self, expr):
        if self.memo[expr] is not None:
            return self.memo[expr]
        self._check_children(expr)
        children = [self.get(a) for a in expr._args]
        new_expr = dex.ProductExpression(children)
        self.set(expr, new_expr)

    def visit_division(self, expr):
        if self.memo[expr] is not None:
            return self.memo[expr]

        self._check_children(expr)
        children = [self.get(a) for a in expr._args]
        new_expr = dex.DivisionExpression(children)
        self.set(expr, new_expr)

    def visit_sum(self, expr):
        if self.memo[expr] is not None:
            return self.memo[expr]

        self._check_children(expr)
        children = [self.get(a) for a in expr._args]
        new_expr = dex.SumExpression(children)
        self.set(expr, new_expr)

    def visit_linear(self, expr):
        if self.memo[expr] is not None:
            return self.memo[expr]

        self._check_children(expr)
        children = [self.get(a) for a in expr._args]
        coeffs = [expr._coef[id(a)] for a in expr._args]
        const = expr._const
        new_expr = dex.LinearExpression(
            coeffs, children, const
        )
        self.set(expr, new_expr)

    def visit_negation(self, expr):
        if self.memo[expr] is not None:
            return self.memo[expr]

        self._check_children(expr)
        children = [self.get(a) for a in expr._args]
        new_expr = dex.NegationExpression(children)
        self.set(expr, new_expr)

    def visit_unary_function(self, expr):
        if self.memo[expr] is not None:
            return self.memo[expr]

        self._check_children(expr)
        children = [self.get(a) for a in expr._args]
        assert len(children) == 1
        fun = expr.name
        ExprClass = {
            'sqrt': dex.SqrtExpression,
            'exp': dex.ExpExpression,
            'log': dex.LogExpression,
            'sin': dex.SinExpression,
            'cos': dex.CosExpression,
            'tan': dex.TanExpression,
            'asin': dex.AsinExpression,
            'acos': dex.AcosExpression,
            'atan': dex.AtanExpression,
        }.get(fun)
        if ExprClass is None:
            raise AssertionError('Unknwon function', fun)
        new_expr = ExprClass(children)
        self.set(expr, new_expr)


    def visit_abs(self, expr):
        if self.memo[expr] is not None:
            return self.memo[expr]

        self._check_children(expr)
        children = [self.get(a) for a in expr._args]
        new_expr = dex.AbsExpression(children)
        self.set(expr, new_expr)

    def visit_pow(self, expr):
        if self.memo[expr] is not None:
            return self.memo[expr]

        self._check_children(expr)
        children = [self.get(a) for a in expr._args]
        new_expr = dex.PowExpression(children)
        self.set(expr, new_expr)


class ComponentFactory(object):
    def __init__(self, dag):
        self._components = ExpressionDict(float_hasher=BTreeFloatHasher())
        self.dag = dag

    def variable(self, omo_var):
        comp = self._components.get(omo_var)
        if comp is not None:
            return comp
        domain = convert_domain(omo_var.domain)
        new_var = dex.Variable(omo_var.name, omo_var.lb, omo_var.ub, domain)
        self._components[omo_var] = new_var
        return new_var

    def constraint(self, omo_cons):
        bounds, expr = bounds_and_expr(omo_cons.expr)
        new_expr = self.expression(expr)
        constraint = dex.Constraint(
            omo_cons.name,
            bounds.lower,
            bounds.upper,
            [new_expr],
        )
        new_expr.add_parent(constraint)
        return constraint

    def objective(self, omo_obj):
        if omo_obj.is_minimizing():
            sense = dex.Sense.MINIMIZE
        else:
            sense = dex.Sense.MAXIMIZE

        new_expr = self.expression(omo_obj.expr)
        obj = dex.Objective(
            omo_obj.name, sense=sense, children=[new_expr]
        )
        new_expr.add_parent(obj)
        return obj

    def expression(self, expr):
        return convert_expression(self._components, self.dag, expr)
