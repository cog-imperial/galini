# pylint: skip-file
import pytest
import numpy as np
from galini.core import *
from suspect.interval import Interval


def test_foo():
    x = Interval(0, 1)
    y = Interval(1, 2)
    s = PowExpression([0, 1])
    x = s.eval(np.array([x, y]))
    y = s.d_v(0, np.array([x, y]))
    z = s.dd_vv(0, 1, np.array([x, y]))
    print(x, y, z)
    assert False
