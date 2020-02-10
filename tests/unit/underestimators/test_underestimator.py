# pylint: skip-file
import pytest
from galini.core import Constraint, Variable
from galini.expression_relaxation import ExpressionRelaxationResult


class Foo:
    pass


class TestUnderestimatorResult:
    def test_expression_must_be_expression(self):
        a = Foo()
        with pytest.raises(ValueError):
            ExpressionRelaxationResult(a)

    def test_constraints_must_be_constraints(self):
        v = Variable('v', None, None, None)
        c = Constraint('foo', v, None, None)
        with pytest.raises(ValueError):
            ExpressionRelaxationResult(v, [c, Foo()])
