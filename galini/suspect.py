# Copyright 2018 Francesco Ceccon
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""GALINI and SUSPECT integration."""
from suspect.interfaces import ForwardIterator, BackwardIterator


class ExpressionDict(object):
    def __init__(self, problem):
        self._dict = [None] * problem.size

    def get(self, key, default=None):
        result =  self._dict[key.idx]
        if result is None:
            return default
        return result

    def __getitem__(self, item):
        return self._dict[item.idx]

    def __call__(self, item):
        return self._dict[item.idx]

    def __setitem__(self, key, value):
        self._dict[key.idx] = value

    def __contains__(self, item):
        return self._dict[item.idx] is not None

    def __len__(self):
        return len(self._dict)


class ProblemContext(object):
    def __init__(self, problem, bounds=None, monotonicity=None, convexity=None):
        if bounds is None:
            bounds = ExpressionDict(problem)
        if monotonicity is None:
            monotonicity = ExpressionDict(problem)
        if convexity is None:
            convexity = ExpressionDict(problem)
        self.bounds = bounds
        self.polynomial = ExpressionDict(problem)
        self.monotonicity = monotonicity
        self.convexity = convexity
        self.metadata = {}

    def get_bounds(self, expr):
        return self.bounds[expr]

    def set_bounds(self, expr ,value):
        self.bounds[expr] = value

    def set_polynomiality(self, expr, value):
        self.polynomial[expr] = value

    def set_monotonicity(self, expr, value):
        self.monotonicity[expr] = value

    def set_convexity(self, expr, value):
        self.convexity[expr] = value


class ProblemForwardIterator(ForwardIterator):
    def iterate(self, problem, visitor, *args, **_kwargs):
        changed_vertices = []
        for vertex in problem.vertices:
            has_changed = visitor.visit(vertex, *args)
            if has_changed:
                changed_vertices.append(vertex)
        return changed_vertices


class ProblemBackwardIterator(BackwardIterator):
    def iterate(self, problem, visitor, *args, **kwargs):
        changed_vertices = set()
        starting_vertices = kwargs.pop('starting_vertices', None)
        if starting_vertices is None:
            vertices = [con.root_expr for con in problem.constraints]
            vertices.append(problem.objective.root_expr)
        else:
            vertices = starting_vertices

        while vertices:
            vertex = vertices.pop()
            has_changed = visitor.visit(vertex, *args)
            if has_changed:
                for child in vertex.children:
                    changed_vertices.add(child)
                    vertices.append(child)
        return list(changed_vertices)
