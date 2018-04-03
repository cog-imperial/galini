# pylint: disable=invalid-name
"""Arbitrary precision mathematical constants and comparison."""
import mpmath


mpf = mpmath.mpf
inf = mpf('inf')
pi = mpmath.pi
sin = mpmath.sin
cos = mpmath.cos
sqrt = mpmath.sqrt
log = mpmath.log
exp = mpmath.exp
sin = mpmath.sin
asin = mpmath.asin
cos = mpmath.cos
acos = mpmath.acos
tan = mpmath.tan
atan = mpmath.atan
isnan = mpmath.isnan


def almosteq(a, b):
    """Floating point equality check between `a` and `b`."""
    if abs(a) == inf and abs(b) == inf:
        return True
    return mpmath.almosteq(a, b)


def almostgte(a, b):
    """Return True if a >= b."""
    return a > b or almosteq(a, b)


def almostlte(a, b):
    """Return True if a <= b."""
    return a < b or almosteq(a, b)
