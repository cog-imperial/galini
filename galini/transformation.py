# Copyright 2019 Francesco Ceccon
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

"""Apply transformations to GALINI expressions."""
import abc


class TransformationResult(object):
    """Return value of a `Transformation`.

    Contains a `new_expr` that replaces the original expression,
    and a list of `new_constraints` to add to the problem.
    """
    def __init__(self, new_expr, new_constraints):
        self.expression = new_expr
        self.constraints = new_constraints


class Transformation(metaclass=abc.ABCMeta):
    """Transform an expression of `source_problem` to a new one."""
    def __init__(self, source_problem, target_problem):
        self.source = source_problem
        self.target = target_problem

    @abc.abstractmethod
    def apply(self, expr, ctx):
        """Apply transformation to `expr`."""
        pass
