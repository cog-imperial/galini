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
"""Pyomo reader module."""
import os
import importlib
import importlib.util
from suspect.pyomo.osil_reader import read_osil
from suspect.pyomo.qplib_reader import read_qplib
from galini.error import (
    InvalidFileExtensionError,
    InvalidPythonInputError,
)


def read_python(filename, **_kwargs):
    """Read Pyomo model from Python file.

    Arguments
    ---------
    filename : str
        the input file.

    Returns
    -------
    ConcreteModel
        Pyomo concrete model.
    """
    spec = importlib.util.spec_from_file_location('_input_model_module', filename)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if not hasattr(module, 'get_pyomo_model'):
        raise InvalidPythonInputError('invalid python input. Missing get_pyomo_model function')
    return module.get_pyomo_model()


READER_BY_EXT = {
    '.osil': read_osil,
    '.xml': read_osil,
    '.qplib': read_qplib,
    '.py': read_python,
}


def read_pyomo_model(filename, **kwargs):
    """Read Pyomo model from file.

    Arguments
    ---------
    filename : str
        the input file.

    Returns
    -------
    ConcreteModel
        Pyomo concrete model.
    """
    _, ext = os.path.splitext(filename)
    if ext not in READER_BY_EXT:
        raise InvalidFileExtensionError('invalid extension')
    reader = READER_BY_EXT[ext]
    return reader(filename, **kwargs)
