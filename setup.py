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

# pylint: skip-file
from pathlib import Path
from setuptools import setup, find_packages
from setuptools.extension import Extension
from Cython.Build import cythonize

import numpy as np

project_root = Path(__file__).resolve().parent

about = {}
version_path = project_root / 'galini' / '__version__.py'
with version_path.open() as f:
    exec(f.read(), about)

extensions = [
    Extension(
        'galini.core.dag',
        sources=['galini/core/dag.pyx'],
        include_dirs=[np.get_include()],
    ),
    Extension(
        'galini.core.ad',
        sources=['galini/core/ad.pyx'],
        include_dirs=[np.get_include()],
    ),
]

setup(
    name='galini',
    description=about['__description__'],
    author=about['__author__'],
    author_email=about['__author_email__'],
    licens=about['__license__'],
    version=about['__version__'],
    packages=find_packages(exclude=['tests']),
    entry_points={
        'console_scripts': [
            'galini=galini.cli:main',
        ],
        'galini.subcommands': [
            'solve=galini.subcommands.solve:SolveCommand',
            'dot=galini.subcommands.dot:DotCommand',
        ],
        'galini.solvers': [
            'oa=galini.solvers.outer_approximation:OuterApproximationSolver',
            'ipopt=galini.nlp:IpoptNLPSolver',
        ],
        'galini.nlp_solvers': [
            'ipopt=galini.nlp:IpoptNLPSolver'
        ],
    },
    ext_modules=cythonize(extensions, annotate=True),
    requires=['pyomo'],
    setup_requires=['pytest-runner', 'cython'],
    tests_require=['pytest', 'pytest-cov', 'hypothesis'],
)
