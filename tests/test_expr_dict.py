import pytest
import pyomo.environ as aml
from galini.pyomo.expr_dict import *
from galini.float_hash import RoundFloatHasher
from tests.fixtures import model


def test_expr_hash_linear_expr(model):
    model.cons1 = aml.Constraint(
        rule=lambda m: sum(m.x[i] for i in m.I) == 2
    )
    model.cons2 = aml.Constraint(
        rule=lambda m: sum(m.x[i] for i in m.I) == 2
    )
    model.cons3 = aml.Constraint(
        rule=lambda m: sum(m.x[i] for i in m.I[:2]) == 3
    )

    hash_cons1 = expr_hash(model.cons1.expr)
    hash_cons2 = expr_hash(model.cons2.expr)
    hash_cons3 = expr_hash(model.cons3.expr)

    assert hash_cons1 == hash_cons2
    # TODO: fix too common collisions
    # assert hash_cons1 != hash_cons3

    # we want hashing of linear expr be associative
    model.cons4 = aml.Constraint(
        rule=lambda m: sum(m.x[i] for i in m.I[::-1]) == 2
    )

    hash_cons4 = expr_hash(model.cons4.expr)
    assert hash_cons1 == hash_cons4


def test_expr_hash_product_expr(model):
    model.cons1 = aml.Constraint(
        model.I,
        rule=lambda m, i: 0 <= m.x[i] * m.y[i, 0] <= 10
    )

    hash_cons1 = expr_hash(model.cons1[0].expr)
    for i in model.I:
        for j in model.J:
            if i == 0 and j == 0:
                continue
            # Collisions could happen, so a failure here is expected
            # in some rare cases
            ex = -10 <= model.x[i] * model.y[i, j] <= 100
            assert hash_cons1 != expr_hash(ex)


def test_expr_hash_unary_expr(model):
    model.cons1 = aml.Constraint(
        model.I,
        rule=lambda m, i: aml.cos(m.x[i]) <= 0.5
    )

    hash_cons1 = expr_hash(model.cons1[2].expr)
    ex1 = aml.cos(model.x[2]) <= 0.5
    assert hash_cons1 == expr_hash(ex1)


def test_expr_hash_float_hasher(model):
    model.cons1 = aml.Constraint(
        rule=lambda m: sum(m.x[i] for i in m.I) == 2
    )
    model.cons2 = aml.Constraint(
        rule=lambda m: sum((i+1)*m.x[i] for i in m.I) == 2
    )

    h = RoundFloatHasher(2)

    assert expr_hash(model.cons1.expr, h) != expr_hash(model.cons2.expr, h)
    assert expr_hash(model.cons1.expr) != expr_hash(model.cons2.expr, h)

    model.cons3 = aml.Constraint(
        model.I,
        rule=lambda m, i: m.x[i] == i
    )
    model.cons4 = aml.Constraint(
        model.I,
        rule=lambda m, i: m.x[i] == (i + 1)
    )

    assert expr_hash(model.cons3[0], h) != expr_hash(model.cons4[0], h)


def test_expr_equal(model):
    assert not expr_equal(1.2, 2.3)
    assert expr_equal(model.x[1], model.x[1])
    assert expr_equal(model.x[1] + 1.23, model.x[1] + 1.23)
    assert not expr_equal(model.x[1], model.y[1, 0])
    assert not expr_equal(model.x[1], 2)

    sum1 = sum(model.x[i] for i in model.I)
    sum2 = sum(model.x[i] for i in model.I)
    sum3 = sum((i + 1) * model.x[i] for i in model.I)
    sum4 = sum(i * model.x[i] for i in model.I)
    sum5 = sum((i + 1) * model.x[i] for i in model.I)
    sum6 = sum(model.y[i, 0] for i in model.I)

    assert expr_equal(sum1, sum2)
    assert not expr_equal(sum1, sum3)
    assert not expr_equal(sum1, sum4)
    assert expr_equal(sum3, sum5)
    assert not expr_equal(sum1, sum6)

    fun1 = aml.cos(sum1)
    fun2 = aml.cos(sum2)
    fun3 = aml.sin(sum1)

    assert expr_equal(fun1, fun2)
    assert not expr_equal(fun1, fun3)
    assert expr_equal(fun1 * fun3, fun2 * fun3)
    assert expr_equal(fun1 / fun3, fun2 / fun3)


def test_expr_map(model):
    model.cons1 = aml.Constraint(
        rule=lambda m: sum(m.x[i] for i in m.I) == 2.34
    )
    model.cons2 = aml.Constraint(
        rule=lambda m: sum((i+1)*m.x[i] for i in m.I) == 3
    )

    memo = ExpressionDict()

    assert memo[model.cons1.expr] is None

    memo[model.cons1.expr] = 42
    assert 42 == memo[model.cons1.expr]

    assert memo[model.cons2.expr] is None

    memo[model.cons2.expr] = 123
    assert 123 == memo[model.cons2.expr]

    memo[model.cons1.expr] = 100
    assert 100 == memo[model.cons1.expr]

    assert len(memo) == 2
