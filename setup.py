from setuptools import setup, find_packages
from setuptools.extension import Extension
from Cython.Build import cythonize

import numpy as np

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
    description='',
    author='Francesco Ceccon',
    author_email='francesco.ceccon14@imperial.ac.uk',
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
