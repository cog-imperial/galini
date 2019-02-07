# pylint: skip-file
import pytest
import numpy as np
from galini.core import *
from suspect.expression import ExpressionType


class TestLinearExpression:
    def test_coefficient(self):
        x = Variable('x', None, None, Domain.REAL)
        y = Variable('y', None, None, Domain.REAL)
        z = Variable('z', None, None, Domain.REAL)

        linear = LinearExpression([x, y, z], [1.0, 2.0, 3.0], -5.0)
        assert linear.coefficient(x) == 1.0
        assert linear.coefficient(y) == 2.0
        assert linear.coefficient(z) == 3.0

    def test_linear_of_linear(self):
        x = Variable('x', None, None, Domain.REAL)
        y = Variable('y', None, None, Domain.REAL)
        z = Variable('z', None, None, Domain.REAL)

        # x + 2y + 10.0
        linear1 = LinearExpression([x, y], [1.0, 2.0], 10.0)

        # y + 3z - 5.0
        linear2 = LinearExpression([y, z], [1, 3.0], -5.0)

        linear = LinearExpression([linear1, linear2])
        assert linear.expression_type == ExpressionType.Linear
        assert len(linear.children) == 3
        assert linear.coefficient(x) == 1.0
        assert linear.coefficient(y) == 3.0
        assert linear.coefficient(z) == 3.0
        assert linear.constant_term == 5.0


class TestQuadraticExpression:
    def test_coefficient(self):
        x = Variable('x', None, None, Domain.REAL)
        y = Variable('y', None, None, Domain.REAL)
        z = Variable('z', None, None, Domain.REAL)

        # xy + 2*xz + 3*yz + 4y**2
        quadratic = QuadraticExpression([x, x, y, y], [y, z, z, y], [1.0, 2.0, 3.0, 4.0])
        assert quadratic.coefficient(x, y) == 1.0
        assert quadratic.coefficient(z, x) == 2.0
        assert quadratic.coefficient(z, y) == 3.0
        assert quadratic.coefficient(y, y) == 4.0
