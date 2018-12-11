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

    def __getitem__(self, item):
        return self._dict[item.idx]

    def __call__(self, item):
        return self._dict[item.idx]

    def __setitem__(self, key, value):
        self._dict[key.idx] = value


class ProblemContext(object):
    def __init__(self, problem):
        self.bounds = ExpressionDict(problem)
        self.polynomial = ExpressionDict(problem)
        self.monotonicity = ExpressionDict(problem)
        self.convexity = ExpressionDict(problem)

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
    def iterate(self, problem, visitor, ctx, *_args, **_kwargs):
        changed_vertices = []
        for vertex in problem.vertices:
            has_changed = visitor.visit(vertex, ctx)
            if has_changed:
                changed_vertices.append(vertex)
        return changed_vertices


class ProblemBackwardIterator(BackwardIterator):
    def iterate(self, problem, visitor, ctx, *args, **kwargs):
        changed_vertices = []
        for vertex in reversed(problem.vertices):
            has_changed = visitor.visit(vertex, ctx)
            if has_changed:
                changed_vertices.append(vertex)
        return changed_vertices
