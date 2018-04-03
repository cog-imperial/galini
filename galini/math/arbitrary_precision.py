import sys
import mpmath


mpf = mpmath.mpf
inf = mpf('inf')

# Re-export symbols from mpmath
_EXPORTED_SYMBOLS = [
    'pi', 'sin', 'cos', 'sqrt', 'log', 'sin', 'asin', 'cos', 'acos',
    'tan', 'atan', 'exp']

_module = sys.modules[__name__]
for sym in _EXPORTED_SYMBOLS:
    setattr(_module, sym, getattr(mpmath, sym))


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
