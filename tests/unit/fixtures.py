import pytest
import pyomo.environ as aml
import hypothesis.strategies as st
from galini.core import Variable


@pytest.fixture
def model():
    m = aml.ConcreteModel()
    m.I = range(10)
    m.J = range(5)
    m.x = aml.Var(m.I)
    m.y = aml.Var(m.I, m.J, domain=aml.NonNegativeReals, bounds=(0, 1))
    m.z = aml.Var(m.I, bounds=(0, 10))
    return m


@st.composite
def variables(draw):
    domain = draw(st.sampled_from(
        Domain.REALS,
        Domain.INTEGERS,
        Domain.BINARY,
        ))
    lower_bound = draw(st.one_of(st.none(), st.floats()))
    upper_bound = draw(st.one_of(st.none(), st.floats(min_value=lower_bound)))
    return Variable(lower_bound, upper_bound, domain)
