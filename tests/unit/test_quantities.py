# pylint: skip-file
import pytest
from hypothesis import given
import hypothesis.strategies as st
import numpy as np
from galini.quantities import absolute_gap, relative_gap


class TestAbsoluteGap:
    @given(st.floats(allow_nan=False, allow_infinity=False, max_value=1e20),
           st.floats(min_value=0.0, max_value=1e20, allow_infinity=False))
    def test_absolute_gap_of_finite_numbers(self, lb, expected_gap):
        ub = lb + expected_gap
        gap = absolute_gap(lb, ub)
        if np.isclose(ub, 0.0):
            relative_error = np.abs(expected_gap - gap) / 1e-5
        else:
            relative_error = np.abs(expected_gap - gap) / np.abs(ub)
        assert relative_error < 1e-5

    @given(st.floats(allow_nan=False))
    def test_absolute_gap_of_infinite_upper_bounds(self, lb):
        gap = absolute_gap(lb, np.inf)
        assert gap == np.inf

    @given(st.floats(allow_nan=False))
    def test_absolute_gap_of_infinite_lower_bounds(self, ub):
        gap = absolute_gap(-np.inf, ub)
        assert gap == np.inf


class TestRelativeGap:

    def test_relative_gap_of_finite_numbers(self):
        lb = 100.0
        ub = 100.00001
        gap = relative_gap(lb, ub)
        assert gap < 1e-5

    @given(st.floats(allow_nan=False))
    def test_relative_gap_of_infinite_upper_bounds(self, lb):
        gap = relative_gap(lb, np.inf)
        assert gap == np.inf

    @given(st.floats(allow_nan=False))
    def test_relative_gap_of_infinite_lower_bounds(self, ub):
        gap = relative_gap(-np.inf, ub)
        assert gap == np.inf
