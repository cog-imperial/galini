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

import os
import subprocess
import sys
from distutils.cmd import Command
from distutils.spawn import find_executable
from pathlib import Path

from setuptools import setup, find_packages
from setuptools.command.test import test as TestCommand

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
            'special_structure=galini.commands.special_structure:SpecialStructureCommand',
            'info=galini.commands.info:InfoCommand',
            'plugins=galini.commands.plugins:PluginsCommand',
        ],
        'galini.algorithms': [
            'bac=galini.branch_and_cut:BranchAndCutAlgorithm',
        ],
        'galini.initial_primal_search': [
            'default=galini.branch_and_cut.primal:DefaultInitialPrimalSearchStrategy',
            'no_primal=galini.branch_and_cut.primal:NoInitialPrimalSearchStrategy',
        ],
        'galini.primal_heuristic': [
            'default=galini.branch_and_cut.primal:DefaultPrimalHeuristic',
        ],
        'galini.branching_strategy': [
            'default=galini.branch_and_cut.branching:BranchAndCutBranchingStrategy',
        ],
        'galini.node_selection_strategy': [
            'default=galini.branch_and_bound.selection:BestLowerBoundSelectionStrategy',
        ],
        'galini.relaxation': [
            'default=galini.branch_and_cut.relaxation:DefaultRelaxation',
        ],
        'galini.cuts_generators': [
            'outer_approximation=galini.outer_approximation:OuterApproximationCutsGenerator',
            'triangle=galini.triangle:TriangleCutsGenerator',
            'sdp=galini.sdp:SdpCutsGenerator',
        ],
    },
    cmdclass={
        'test': PyTestCommand,
        'generate_proto': GenerateProto,
    },
    include_package_data=True,
    install_requires=[
        'pyomo>=5.6.7',
        'cog-suspect>=2.1.2',
        'numpy>=1.15',
        'toml',
        'pydot',
        'texttable>=1.4.0',
        'pytimeparse>=1.1.8',
        'protobuf>=3.0',
        'h5py>=2.0',
        'networkx>=2.4',
        'coramin>=0.1.0',
        'scipy>=1.5.2',
    ],
    setup_requires=['pytest-runner'],
    tests_require=[
        'pytest',
        'pytest-cov',
        'hypothesis',
        'pytest-benchmark',
    ],
)
