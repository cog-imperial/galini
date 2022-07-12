#  Copyright 2020 Francesco Ceccon
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.

import os

import pyomo.environ as pyo

from galini.pyomo.qplib_reader import read_qplib


def test_qplib_reader_on_example():
    example_file = os.path.join(
        os.path.dirname(__file__),
        './fixtures/example.qplib'
    )
    model = read_qplib(example_file)
    assert model.name == 'MIPBAND'
    assert model.x1.domain == pyo.Reals
    assert model.x2.domain == pyo.Reals
    assert model.x3.domain == pyo.Binary
    assert len(list(model.component_data_objects(pyo.Constraint))) == 2