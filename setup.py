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

import glob
import os
import sys
import subprocess
from distutils.cmd import Command
from distutils.spawn import find_executable
from pathlib import Path

from setuptools import setup, find_packages
from setuptools.command.build_ext import build_ext
from setuptools.command.test import test as TestCommand
from setuptools.extension import Extension

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
        ('pdb', None, 'Drop into PDB on failure.'),
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
        self.pdb = None

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

        if self.pdb:
            self.pytest_args.append('--pdb')

        errno = pytest.main(self.pytest_args)
        return sys.exit(errno)



if 'PROTOC' in os.environ and os.path.exists(os.environ['PROTOC']):
    protoc = os.environ['PROTOC']
else:
    protoc = find_executable('protoc')


def generate_proto(source):
    output = source.replace('.proto', '_pb2.py')

    print('Generating {}...'.format(output))

    if not os.path.exists(source):
        print('Protobuf file {} does not exist.'.format(source), file=sys.stderr)
        sys.exit(-1)

    if protoc is None:
        print('protoc executable not found. Please install it.', file=sys.stderr)
        sys.exit(-1)

    command = [protoc, '-Iproto', '--python_out=galini/io', source]
    if subprocess.call(command) != 0:
        sys.exit(-1)


class GenerateProto(Command):
    description = 'Generate _pb2.py files from .proto definition.'
    user_options = []

    def initialize_options(self):
        pass

    def finalize_options(self):
        pass

    def run(self):
        generate_proto('proto/message.proto')


class get_pybind11_include(object):
    def __init__(self, user=False):
        self.user = user

    def __str__(self):
        if os.environ.get('PYBIND11_INCLUDE_DIR'):
            return os.environ.get('PYBIND11_INCLUDE_DIR')

        import pybind11
        return pybind11.get_include(self.user)


def get_ipopt_include():
    if os.environ.get('IPOPT_INCLUDE_DIR'):
        return os.environ.get('IPOPT_INCLUDE_DIR')


def get_ipopt_library_dir():
    if os.environ.get('IPOPT_LIBRARY_DIR'):
        return os.environ.get('IPOPT_LIBRARY_DIR')


def get_include_dirs():
    dirs = ['src', get_pybind11_include(), get_pybind11_include(user=True), get_ipopt_include()]
    return [d for d in dirs if d is not None]


def get_library_dirs():
    dirs = [get_ipopt_library_dir()]
    return [d for d in dirs if d is not None]


extensions = [
    Extension(
        'galini_core',
        sources=[
            'src/uid.cc',
            'src/ad/values.cc',
            'src/ad/ad_adapter.cc',
            'src/expression/expression_base.cc',
            'src/expression/auxiliary_variable.cc',
            'src/expression/unary_function_expression.cc',
            'src/expression/binary_expression.cc',
            'src/expression/nary_expression.cc',
            'src/expression/graph.cc',
            'src/problem/problem_base.cc',
            'src/problem/root_problem.cc',
            'src/problem/child_problem.cc',
            'src/ipopt/ipopt_solve.cc',
            'src/ad/module.cc',
            'src/expression/module.cc',
            'src/problem/module.cc',
            'src/ipopt/module.cc',
            'src/core.cc',
        ],
        include_dirs=get_include_dirs(),
        library_dirs=get_library_dirs(),
        language='c++',
        depends=glob.glob('src/**/*.h'),
        libraries=['ipopt'],
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
            'plugins=galini.commands.plugins:PluginsCommand',
            'abb=galini.commands.abb:AbbCommand',
        ],
        'galini.solvers': [
            'mip=galini.mip.solver:MIPSolver',
            'ipopt=galini.ipopt:IpoptNLPSolver',
            'slsqp=galini.slsqp:SlsqpSolver',
            'bac=galini.branch_and_cut:BranchAndBoundSolver',
        ],
        'galini.cuts_generators': [
            'triangle=galini.triangle:TriangleCutsGenerator',
            'mixed_integer_outer_approximation=galini.outer_approximation:MixedIntegerOuterApproximationCutsGenerator',
            'outer_approximation=galini.outer_approximation:OuterApproximationCutsGenerator',
            'sdp=galini.sdp:SdpCutsGenerator',
        ],
    },
    ext_modules=extensions,
    cmdclass={
        'test': PyTestCommand,
        'build_ext': BuildExt,
        'generate_proto': GenerateProto,
    },
    include_package_data=True,
    install_requires=[
        'pyomo>=5.6.7',
        'cog-suspect>=1.6.5',
        'pulp>=1.6',
        'numpy',
        'toml',
        'pydot',
        'texttable>=1.4.0',
        'pybind11>=2.2.3',
        'pytimeparse>=1.1.8',
    ],
    setup_requires=['pytest-runner', 'pybind11'],
    tests_require=[
        'pytest',
        'pytest-cov',
        'hypothesis',
        'pytest-benchmark',
    ],
)
