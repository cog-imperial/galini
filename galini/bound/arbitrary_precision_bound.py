from numbers import Number
from galini.error import DomainError
from galini.math.arbitrary_precision import (
    inf,
    mpf,
    isnan,
    almostlte,
    almostgte,
    almosteq,
    pi,
    sin,
    tan,
    atan,
    asin,
    acos,
    log,
    exp,
    sqrt,
)
from galini.bound.bound import Bound


class ArbitraryPrecisionBound(Bound):
    def __init__(self, lower, upper):
        if lower is None:
            lower = -inf
        if upper is None:
            upper = inf

        lower = mpf(lower)
        upper = mpf(upper)

        if lower > upper:
            raise ValueError('lower must be <= upper')

        self.lower = lower
        self.upper = upper

    @property
    def lower_bound(self):
        return self.lower

    @property
    def upper_bound(self):
        return self.upper

    def is_zero(self):
        return almosteq(self.lower, 0) and almosteq(self.upper, 0)

    def is_positive(self):
        return self.lower > 0

    def is_negative(self):
        return self.upper < 0

    def is_nonpositive(self):
        return almostlte(self.upper, 0)

    def is_nonnegative(self):
        return almostgte(self.lower, 0)

    def tighten(self, other):
        if not isinstance(other, ArbitraryPrecisionBound):
            raise AssertionError('other must be ArbitraryPrecisionBound')

        if other.lower < self.lower:
            new_l = self.lower
        else:
            new_l = other.lower

        if other.upper > self.upper:
            new_u = self.upper
        else:
            new_u = other.upper

        if new_u < new_l:
            raise DomainError(
                "Invalid tightened bound. This probably means the "
                "resulting domain is empty. This is not currently "
                "supported as it should not happen."
            )

        return ArbitraryPrecisionBound(new_l, new_u)

    def add(self, other):
        l = self.lower
        u = self.upper
        if isinstance(other, ArbitraryPrecisionBound):
            return ArbitraryPrecisionBound(l + other.lower, u + other.upper)
        elif isinstance(other, Number):
            if mpf(other) == inf:
                raise AssertionError('Infinity constants are not allowed')
            return ArbitraryPrecisionBound(l + other, u + other)
        else:
            raise TypeError(
                "adding ArbitraryPrecisionBound to incompatbile type"
            )

    def sub(self, other):
        l = self.lower
        u = self.upper
        if isinstance(other, ArbitraryPrecisionBound):
            return ArbitraryPrecisionBound(l - other.upper, u - other.lower)
        elif isinstance(other, Number):
            if mpf(other) == inf:
                raise AssertionError('Infinity constants are not allowed')
            return ArbitraryPrecisionBound(l - other, u - other)
        else:
            raise TypeError(
                "subtracting ArbitraryPrecisionBound to incompatbile type"
            )

    def mul(self, other):
        l = self.lower
        u = self.upper
        if isinstance(other, ArbitraryPrecisionBound):
            # Check for zero to handle case we or other has an
            # infinite bound.
            if self.is_zero() or other.is_zero():
                return self.zero()
            ol = other.lower
            ou = other.upper
            # 0*inf returns nan. We want to avoid it so that we can
            # return reasonable values
            candidates = [
                c for c in [l*ol, l*ou, u*ol, u*ou]
                if not isnan(c)
            ]
            new_l = min(candidates)
            new_u = max(candidates)
            return ArbitraryPrecisionBound(new_l, new_u)
        elif isinstance(other, Number):
            if mpf(other) == inf:
                raise AssertionError('Infinity constants are not allowed')
            return self.__mul__(ArbitraryPrecisionBound(other, other))
        else:
            raise TypeError(
                "multiplying ArbitraryPrecisionBound to incompatible type"
            )

    def div(self, other):
        if isinstance(other, ArbitraryPrecisionBound):
            ol = other.lower
            ou = other.upper
            if almostlte(ol, 0) and almostgte(ou, 0):
                return ArbitraryPrecisionBound(-inf, inf)
            else:
                return self.__mul__(ArbitraryPrecisionBound(1/ou, 1/ol))
        elif isinstance(other, Number):
            if mpf(other) == inf:
                raise AssertionError('Infinity constants are not allowed')
            return self.__truediv__(ArbitraryPrecisionBound(other, other))
        else:
            raise TypeError(
                "dividing AribtraryPrecisionBound by incompatible type"
            )

    def equals(self, other):
        if not isinstance(other, ArbitraryPrecisionBound):
            return False
        return (
            almosteq(self.lower, other.lower) and
            almosteq(self.upper, other.upper)
        )

    def contains(self, other):
        if isinstance(other, Number):
            return (
                almostgte(other, self.lower) and
                almostlte(other, self.upper)
            )
        elif isinstance(other, ArbitraryPrecisionBound):
            return (
                almostgte(other.lower, self.lower) and
                almostlte(other.upper, self.upper)
            )
        else:
            raise TypeError(
                "comparing ArbitraryPrecisionBound by incompatible type"
            )

    @staticmethod
    def zero():
        return ArbitraryPrecisionBound(0, 0)

    def size(self):
        if self.lower == -inf or self.upper == inf:
            return inf
        return self.upper - self.lower

    def _negation(self):
        return -self

    _negation_inv = _negation

    def _abs(self):
        new_upper = max(abs(self.lower), abs(self.upper))
        if 0 in self:
            new_lower = 0.0
        else:
            new_lower = min(abs(self.lower), abs(self.upper))
        return ArbitraryPrecisionBound(new_lower, new_upper)

    def _abs_inv(self):
        assert self.upper >= 0
        return ArbitraryPrecisionBound(-self.upper, self.upper)

    def _sqrt(self):
        return ArbitraryPrecisionBound(sqrt(self.lower), sqrt(self.upper))

    def _sqrt_inv(self):
        sqr = self.upper * self.upper
        return ArbitraryPrecisionBound(-sqr, sqr)

    def _exp(self):
        return ArbitraryPrecisionBound(exp(self.lower), exp(self.upper))

    def _log(self):
        return ArbitraryPrecisionBound(log(self.lower), log(self.upper))

    _exp_inv = _log
    _log_inv = _exp

    def _sin(self):
        if almostgte(self.size(), 2*pi):
            return ArbitraryPrecisionBound(-1, 1)
        else:
            l = self.lower % (2 * pi)
            u = l + (self.upper - self.lower)
            new_u = max(sin(l), sin(u))
            new_l = min(sin(l), sin(u))
            b = ArbitraryPrecisionBound(l, u)
            if 0.5*pi in b:
                new_u = 1
            if 1.5*pi in b:
                new_l = -1
            return ArbitraryPrecisionBound(new_l, new_u)

    def _cos(self):
        if almostgte(self.size(), 2*pi):
            return ArbitraryPrecisionBound(-1, 1)
        else:
            # translate left by pi/2
            pi_2 = pi / mpf('2')
            return (self + pi_2).sin()

    def _tan(self):
        if almostgte(self.size(), pi):
            return ArbitraryPrecisionBound(None, None)
        else:
            l = self.lower_bound % pi
            u = l + (self.upper - self.lower)
            tan_l = tan(l)
            tan_u = tan(u)
            new_l = min(tan_l, tan_u)
            new_u = max(tan_l, tan_u)


            if almosteq(l, 0.5 * pi):
                new_l = None

            if almosteq(u, 0.5 * pi):
                new_u = None

            if new_l is not None and new_u is not None:
                b = ArbitraryPrecisionBound(l, u)
                if 0.5*pi in b or pi in b:
                    return ArbitraryPrecisionBound(None, None)

            return ArbitraryPrecisionBound(new_l, new_u)

    def _asin(self):
        return ArbitraryPrecisionBound(asin(self.lower), asin(self.upper))

    def _acos(self):
        return ArbitraryPrecisionBound(acos(self.upper), acos(self.lower))

    def _atan(self):
        return ArbitraryPrecisionBound(atan(self.lower), atan(self.upper))

    _sin_inv = _asin
    _cos_inv = _acos
    _tan_inv = _atan
    _asin_inv = _sin
    _acos_inv = _cos
    _atan_inv = _tan
