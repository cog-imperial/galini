import pytest
from tests.unit.conftest import PlaceholderExpression
from galini.dag.dag import VerticesList
from hypothesis import given, assume
import hypothesis.strategies as st


class TestVerticesList(object):
    @pytest.mark.parametrize('reverse', [True, False])
    @given(depths=st.lists(st.integers(min_value=0, max_value=100), min_size=1))
    def test_with_starting_vertices(self, depths, reverse):
        vertices = [PlaceholderExpression() for _ in depths]
        for i, v in enumerate(vertices):
            v.depth = depths[i]
        vl = VerticesList(vertices, reverse=reverse)
        assert [v.depth for v in vl] == sorted(depths, reverse=reverse)

    @pytest.mark.parametrize('reverse', [True, False])
    @given(depths=st.lists(st.integers(min_value=0, max_value=100), min_size=1))
    def test_append(self, depths, reverse):
        vl = VerticesList(reverse=reverse)
        for d in depths:
            p = PlaceholderExpression()
            p.depth = d
            vl.append(p)
        assert [v.depth for v in vl] == sorted(depths, reverse=reverse)

    @pytest.mark.parametrize('reverse', [True, False])
    @given(depths=st.lists(st.integers(min_value=0, max_value=100), min_size=1))
    def test_pop(self, depths, reverse):
        vertices = [PlaceholderExpression() for _ in depths]
        for i, v in enumerate(vertices):
            v.depth = depths[i]
        vl = VerticesList(vertices, reverse=reverse)
        if reverse:
            assert vl.pop().depth == max(depths)
        else:
            assert vl.pop().depth == min(depths)
