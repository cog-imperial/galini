import pytest
from unittest.mock import MagicMock
from hypothesis import given, assume
import hypothesis.strategies as st
from galini.bound.tightening import *
import galini.dag.expressions as dex
from galini.bound import ArbitraryPrecisionBound as Bound
from tests.conftest import (
    PlaceholderExpression,
    bound_description_to_bound,
    coefficients,
    reals,
)
