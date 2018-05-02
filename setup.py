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
            'ipopt=galini.nlp:IpoptNLPSolver',
        ],
        'galini.nlp_solvers': [
            'ipopt=galini.nlp:IpoptNLPSolver'
        ],
    },
    requires=['pyomo'],
    setup_requires=['pytest-runner'],
    tests_require=['pytest', 'pytest-cov', 'hypothesis'],
)
