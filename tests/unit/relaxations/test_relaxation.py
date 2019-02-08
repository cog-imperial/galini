# pylint: skip-file
import pytest
import pyomo.environ as aml
from galini.pyomo import dag_from_pyomo_model
from galini.relaxations import Relaxation, RelaxationResult
from galini.core import Constraint, Variable


@pytest.fixture
def problem():
    m = aml.ConcreteModel(name='test_relaxation')

    m.I = range(5)
    m.x = aml.Var(m.I)

    m.obj = aml.Objective(expr=m.x[0])
    m.cons0 = aml.Constraint(expr=sum(m.x[i] for i in m.I) >= 0)
    m.cons1 = aml.Constraint(expr=-2.0*aml.sin(m.x[0]) + aml.cos(m.x[1]) >= 0)
    m.cons2 = aml.Constraint(expr=m.x[1] * m.x[2] >= 0)

    return dag_from_pyomo_model(m)


def test_relaxation_name(problem):
    class MockRelaxation(Relaxation):
        def relaxed_problem_name(self, problem):
            return problem.name + '_relaxed'

        def relax_objective(self, problem, objective):
            return RelaxationResult(objective)

        def relax_constraint(self, problem, constraint):
            return RelaxationResult(constraint)

    r = MockRelaxation()
    relaxed_problem = r.relax(problem)
    assert relaxed_problem.name == 'test_relaxation_relaxed'


def test_relaxation_returning_original_problem(problem):
    class MockRelaxation(Relaxation):
        def relaxed_problem_name(self, problem):
            return problem.name + '_relaxed'

        def relax_objective(self, problem, objective):
            return RelaxationResult(objective)

        def relax_constraint(self, problem, constraint):
            return RelaxationResult(constraint)

    r = MockRelaxation()
    relaxed_problem = r.relax(problem)

    assert len(problem.variables) == len(relaxed_problem.variables)
    for var, rel_var in zip(problem.variables, relaxed_problem.variables):
        assert var.name == rel_var.name
        assert var.lower_bound == rel_var.lower_bound
        assert var.upper_bound == rel_var.upper_bound
        assert var.domain == rel_var.domain
        assert var.idx == rel_var.idx
        assert var.uid != rel_var.uid
    assert len(problem.objectives) == len(relaxed_problem.objectives)
    assert len(problem.constraints) == len(relaxed_problem.constraints)

    original_vertices = problem.vertices
    relaxed_vertices = relaxed_problem.vertices

    assert len(original_vertices) == len(relaxed_vertices)

    # count expression vertices types
    original_count = {}
    relaxed_count = {}

    for expr in original_vertices:
        if expr.expression_type not in original_count:
            original_count[expr.expression_type] = 0
        original_count[expr.expression_type] += 1

    for expr in relaxed_vertices:
        if expr.expression_type not in relaxed_count:
            relaxed_count[expr.expression_type] = 0
        relaxed_count[expr.expression_type] += 1

    assert set(original_count.keys()) == set(relaxed_count.keys())
    for key, count in original_count.items():
        assert count == relaxed_count[key]


def test_relaxation_returning_new_expressions(problem):
    class MockRelaxation(Relaxation):
        def relaxed_problem_name(self, problem):
            return problem.name + '_relaxed'

        def relax_objective(self, problem, objective):
            return RelaxationResult(objective)

        def relax_constraint(self, problem, constraint):
            if constraint.name != 'cons2':
                return RelaxationResult(constraint)
            w = Variable('aux', None, None, None)
            return RelaxationResult(Constraint('aux_cons2', w, None, None))

    r = MockRelaxation()
    relaxed_problem = r.relax(problem)
    assert len(problem.variables) + 1 == len(relaxed_problem.variables)
