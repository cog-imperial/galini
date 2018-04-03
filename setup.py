from setuptools import setup, find_packages


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
        ],
    },
    requires=['pyomo'],
)
