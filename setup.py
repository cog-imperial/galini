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
import sys
from pathlib import Path
import glob
from setuptools import setup, find_packages
from setuptools.extension import Extension
from setuptools.command.test import test as TestCommand
from setuptools.command.build_ext import build_ext
from Cython.Build import cythonize

import numpy as np

project_root = Path(__file__).resolve().parent

about = {}
version_path = project_root / 'galini' / '__version__.py'
with version_path.open() as f:
    exec(f.read(), about)

with (project_root / 'README.rst').open() as f:
    readme = f.read()

with (project_root / 'CHANGELOG.rst').open() as f:
    changelog = f.read()


class PyTestCommand(TestCommand):
    user_options = [
        ('unit', None, 'Specify to run unit tests only.'),
        ('e2e', None, 'Specify to run end to end tests only.'),
    ]

    def initialize_options(self):
        super().initialize_options()
        self.pytest_args = [
            '--cov', 'galini',
            '--cov-report=html',
            '--cov-report=term',
        ]
        self.unit = None
        self.e2e = None

    def run_tests(self):
        import pytest

        if self.unit and self.e2e:
            raise ValueError('Must specify only one of e2e or unit.')

        if self.unit:
            # run unit tests only
            self.pytest_args.append('tests/unit')

        if self.e2e:
            # run e2e tests only
            self.pytest_args.append('tests/e2e')

        errno = pytest.main(self.pytest_args)
        return sys.exit(errno)


class get_pybind11_include(object):
    def __init__(self, user=False):
        self.user = user

    def __str__(self):
        # import pybind11
        # return pybind11.get_include(self.user)
        return '/data/fc714/pybind11/include'


extensions = [
    Extension(
        'galini_core',
        sources=[
            'src/ad/values.cc',
            'src/ad/ad_adapter.cc',
            'src/expression/expression_base.cc',
            'src/expression/unary_function_expression.cc',
            'src/expression/binary_expression.cc',
            'src/expression/nary_expression.cc',
            'src/problem/problem_base.cc',
            'src/problem/root_problem.cc',
            'src/problem/child_problem.cc',
            'src/ad/module.cc',
            'src/expression/module.cc',
            'src/problem/module.cc',
            'src/core.cc',
        ],
        include_dirs=[
            'src',
            get_pybind11_include(),
            get_pybind11_include(user=True),
        ],
        language='c++',
        depends=glob.glob('src/**/*.h'),
    )
]


class BuildExt(build_ext):
    def build_extensions(self):
        ct = self.compiler.compiler_type
        opts = ['-std=c++14']
        if ct == 'unix':
            opts.append('-DVERSION_INFO="%s"' % self.distribution.get_version())
            opts.append('-fvisibility=hidden')

        for ext in self.extensions:
            ext.extra_compile_args = opts
        build_ext.build_extensions(self)


setup(
    name='galini',
    description=about['__description__'],
    author=about['__author__'],
    author_email=about['__author_email__'],
    license=about['__license__'],
    version=about['__version__'],
    long_description=readme + '\n\n' + changelog,
    packages=find_packages(exclude=['tests']),
    entry_points={
        'console_scripts': [
            'galini=galini.cli:main',
        ],
        'galini.commands': [
            'solve=galini.commands.solve:SolveCommand',
            'dot=galini.commands.dot:DotCommand',
            'special_structure=galini.commands.special_structure:SpecialStructureCommand',
            'info=galini.commands.info:InfoCommand',
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
    cmdclass={
        'test': PyTestCommand,
        'build_ext': BuildExt,
    },
    install_requires=[
        'pyomo>=5.2',
        'cog-suspect>=1.0.6',
        'pypopt>=0.4.3',
        'numpy',
        'toml',
        'pydot',
        'texttable>=1.4.0',
        'pybind11>=2.2.3',
    ],
    setup_requires=['pytest-runner', 'cython', 'pybind11'],
    tests_require=['pytest', 'pytest-cov', 'hypothesis'],
)
