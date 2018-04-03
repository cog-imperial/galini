import abc
import warnings


class FunctionOnBound(object):
    def __init__(self, func, inv_func):
        self.func = func
        self.inv_func = inv_func

    def __call__(self):
        return self.func()

    @property
    def inv(self):
        return FunctionOnBound(self.inv_func, self.func)


class Bound(metaclass=abc.ABCMeta):
    @abc.abstractproperty
    def lower_bound(self):  # pragma: no cover
        pass

    @property
    def l(self):  # pragma: no cover
        warnings.warn(
            "Bound.l is deprecated. Use Bound.lower_bound",
            DeprecationWarning,
        )
        return self.lower_bound

    @abc.abstractproperty
    def upper_bound(self):  # pragma: no cover
        pass

    @property
    def u(self):  # pragma: no cover
        warnings.warn(
            "Bound.u is deprecated. Use Bound.upper_bound",
            DeprecationWarning,
        )
        return self.upper_bound

    @abc.abstractmethod
    def is_zero(self):  # pragma: no cover
        pass

    @abc.abstractmethod
    def is_positive(self):  # pragma: no cover
        pass

    @abc.abstractmethod
    def is_negative(self):  # pragma: no cover
        pass

    @abc.abstractmethod
    def tighten(self, other):  # pragma: no cover
        """Returns a new bound which is the tightest intersection of bounds."""
        pass

    @abc.abstractmethod
    def add(self, other):  # pragma: no cover
        pass

    @abc.abstractmethod
    def sub(self, other):  # pragma: no cover
        pass

    @abc.abstractmethod
    def mul(self, other):  # pragma: no cover
        pass

    @abc.abstractmethod
    def div(self, other):  # pragma: no cover
        pass

    @abc.abstractmethod
    def equals(self, other):  # pragma: no cover
        pass

    @abc.abstractmethod
    def contains(self, other):  # pragma: no cover
        pass

    @abc.abstractmethod
    def zero(self):  # pragma: no cover
        pass

    @abc.abstractmethod
    def is_nonpositive(self):  # pragma: no cover
        pass

    @abc.abstractmethod
    def is_nonnegative(self):  # pragma: no cover
        pass

    @property
    def negation(self):  # pragma: no cover
        """Return the bound of -self"""
        return FunctionOnBound(self._negation, self._negation_inv)

    @abc.abstractmethod
    def _negation(self): pass

    @abc.abstractmethod
    def _negation_inv(self): pass

    @property
    def abs(self):  # pragma: no cover
        """Return the bound of |self|"""
        return FunctionOnBound(self._abs, self._abs_inv)

    @abc.abstractmethod
    def _abs(self): pass

    @abc.abstractmethod
    def _abs_inv(self): pass

    @property
    def sqrt(self):  # pragma: no cover
        """Return the bound of sqrt(self)"""
        return FunctionOnBound(self._sqrt, self._sqrt_inv)

    @abc.abstractmethod
    def _sqrt(self): pass

    @abc.abstractmethod
    def _sqrt_inv(self): pass

    @property
    def exp(self):  # pragma: no cover
        """Return the bound of exp(self)"""
        return FunctionOnBound(self._exp, self._exp_inv)

    @property
    def log(self):  # pragma: no cover
        """Return the bound of log(self)"""
        return FunctionOnBound(self._log, self._log_inv)

    @property
    def sin(self):
        """Return the bound of sin(self)"""
        return FunctionOnBound(self._sin, self._sin_inv)

    @property
    def cos(self):
        """Return the bound of cos(self)"""
        return FunctionOnBound(self._cos, self._cos_inv)

    @property
    def tan(self):
        """Return the bound of tan(self)"""
        return FunctionOnBound(self._tan, self._tan_inv)

    @property
    def asin(self):
        """Return the bound of asin(self)"""
        return FunctionOnBound(self._asin, self._asin_inv)

    @property
    def acos(self):
        """Return the bound of acos(self)"""
        return FunctionOnBound(self._acos, self._acos_inv)

    @property
    def atan(self):
        """Return the bound of atan(self)"""
        return FunctionOnBound(self._atan, self._atan_inv)

    def __add__(self, other):  # pragma: no cover
        return self.add(other)

    def __radd__(self, other):  # pragma: no cover
        return self.__add__(other)

    def __sub__(self, other):  # pragma: no cover
        return self.sub(other)

    def __neg__(self):  # pragma: no cover
        return self.zero() - self

    def __mul__(self, other):  # pragma: no cover
        return self.mul(other)

    def __rmul__(self, other):  # pragma: no cover
        return self.__mul__(other)

    def __truediv__(self, other):  # pragma: no cover
        return self.div(other)

    def __eq__(self, other):  # pragma: no cover
        return self.equals(other)

    def __contains__(self, other):  # pragma: no cover
        return self.contains(other)

    def __repr__(self):  # pragma: no cover
        return '<{} at {}>'.format(str(self), id(self))

    def __str__(self):  # pragma: no cover
        return '[{}, {}]'.format(self.lower_bound, self.upper_bound)
