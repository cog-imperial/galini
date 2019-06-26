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

"""Galini Core module."""
from enum import Enum
from collections import namedtuple


__all__ = [
    'Domain', 'Sense', 'Problem', 'Variable', 'Constant', 'Constraint', 'Objective',
    'Expression', 'UnaryExpression', 'BinaryExpression', 'NaryExpression', 'AuxiliaryVariable',
    'ProductExpression', 'DivisionExpression', 'SumExpression', 'PowExpression',
    'LinearExpression', 'UnaryFunctionExpression', 'NegationExpression', 'AbsExpression',
    'SqrtExpression', 'ExpExpression', 'LogExpression', 'SinExpression', 'CosExpression',
    'TanExpression', 'AsinExpression', 'AcosExpression', 'AtanExpression', 'QuadraticExpression',
    'BilinearTermReference', 'ExpressionReference',
    'RootProblem', 'ChildProblem', 'RelaxedProblem', 'VariableView',
    'ipopt_solve', 'IpoptSolution', 'IpoptApplication', 'PythonJournal',
]


class Domain(Enum):
    REAL = 0
    INTEGER = 1
    BINARY = 2

    def is_integer(self):
        return self != self.REAL

    def is_real(self):
        return self == self.REAL

    def is_binary(self):
        return self == self.BINARY


class Sense(Enum):
    MINIMIZE = 0
    MAXIMIZE = 1

    def is_minimization(self):
        return self == self.MINIMIZE

    def is_maximization(self):
        return not self.is_minimization()


BilinearTermReference = namedtuple('BilinearTermReference', ['var1', 'var2'])
ExpressionReference = namedtuple('ExpressionReference', ['expression'])


# pylint: disable=no-name-in-module
from galini_core import (
    Variable,
    AuxiliaryVariable,
    Constant,
    Constraint,
    Objective,
    Expression,
    UnaryExpression,
    BinaryExpression,
    NaryExpression,
    ProductExpression,
    DivisionExpression,
    SumExpression,
    PowExpression,
    LinearExpression,
    QuadraticExpression,
    UnaryFunctionExpression,
    NegationExpression,
    AbsExpression,
    SqrtExpression,
    ExpExpression,
    LogExpression,
    SinExpression,
    CosExpression,
    TanExpression,
    AsinExpression,
    AcosExpression,
    AtanExpression,
    VariableView,
)

# pylint: disable=no-name-in-module
from galini_core import (
    Problem,
    RootProblem,
    ChildProblem,
    RelaxedProblem,

    ipopt_solve,
    IpoptSolution,
    IpoptApplication,
    EJournalLevel,
    Journalist,
    OptionsList,
    PythonJournal,
)
