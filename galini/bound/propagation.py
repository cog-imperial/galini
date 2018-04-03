import operator
from functools import reduce
import galini.dag.expressions as dex
from galini.dag.visitor import ForwardVisitor
from galini.bound import ArbitraryPrecisionBound as Bound


class BoundsPropagationVisitor(ForwardVisitor):
    def register_handlers(self):
        return {
            dex.Variable: self.visit_variable,
            dex.Constant: self.visit_constant,
            dex.Constraint: self.visit_constraint,
            dex.Objective: self.visit_objective,
            dex.ProductExpression: self.visit_product,
            dex.DivisionExpression: self.visit_division,
            dex.LinearExpression: self.visit_linear,
            dex.PowExpression: self.visit_pow,
            dex.SumExpression: self.visit_sum,
            dex.UnaryFunctionExpression: self.visit_unary_function,
        }

    def handle_result(self, expr, new_bound, ctx):
        if expr in ctx.bound:
            # bounds exists, tighten it
            old_bound = ctx.bound[expr]
            new_bound = old_bound.tighten(new_bound)
            has_changed = new_bound != old_bound
        else:
            has_changed = True
        ctx.bound[expr] = new_bound
        return has_changed

    def visit_variable(self, var, _ctx):
        return Bound(var.lower_bound, var.upper_bound)

    def visit_constant(self, const, _ctx):
        return Bound(const.value, const.value)

    def visit_constraint(self, constr, ctx):
        arg = constr.children[0]
        inner = ctx.bound[arg]
        new_lb = max(inner.lower_bound, constr.lower_bound)
        new_ub = min(inner.upper_bound, constr.upper_bound)
        if not new_lb <= new_ub:
            raise RuntimeError('Infeasible bound.')
        return Bound(new_lb, new_ub)

    def visit_objective(self, obj, ctx):
        arg = obj.children[0]
        return ctx.bound[arg]

    def visit_product(self, expr, ctx):
        bounds = [ctx.bound[c] for c in expr.children]
        return reduce(operator.mul, bounds, 1)

    def visit_division(self, expr, ctx):
        top, bot = expr.children
        return ctx.bound[top] / ctx.bound[bot]

    def visit_linear(self, expr, ctx):
        bounds = [
            coef * ctx.bound[c]
            for coef, c in zip(expr.coefficients, expr.children)
        ]
        bounds.append(Bound(expr.constant_term, expr.constant_term))
        return sum(bounds)

    def visit_sum(self, expr, ctx):
        bounds = [ctx.bound[c] for c in expr.children]
        return sum(bounds)

    def visit_pow(self, _expr, _ctx):
        return Bound(None, None)

    def visit_unary_function(self, expr, ctx):
        arg_bound = ctx.bound[expr.children[0]]
        func = getattr(arg_bound, expr.func_name)
        return func()


def propagate_bounds(dag, ctx, starting_vertices=None):
    """Propagate bounds from sources to sinks.

    Arguments
    ---------
    dag: ProblemDag
      the problem
    ctx: SpecialStructurePropagationContext
      the context containing the initial bounds
    starting_vertices: Expression list
      start propagation from these vertices
    """
    visitor = BoundsPropagationVisitor()
    return dag.forward_visit(visitor, ctx, starting_vertices)
