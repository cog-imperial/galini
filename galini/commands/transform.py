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

"""GALINI abb subcommand."""
from galini.commands import CliCommandWithProblem
from galini.transformations.nonlinear import ReplaceNonlinearTransformation
from galini.transformations.insert_tree import insert_tree
from galini.special_structure import detect_special_structure
from galini.util import print_problem


class TransformCommand(CliCommandWithProblem):
    def execute_with_problem(self, problem, args):
        relaxed = problem.make_relaxed('relaxed')
        transformation = ReplaceNonlinearTransformation(problem, relaxed)
        ctx = detect_special_structure(problem)
        memo = {}
        for obj in problem.objectives:
            result = transformation.apply(obj.root_expr, ctx)
            new_expr = insert_tree(problem, relaxed, result.expression, memo)
            relaxed.add_objective(obj.name, new_expr, obj.sense)
            for new_cons in result.constraints:
                new_expr = insert_tree(problem, relaxed, new_cons.root_expr, memo)
                relaxed.add_constraint(
                    new_cons.name,
                    new_expr,
                    new_cons.lower_bound,
                    new_cons.upper_bound,
                )

        for cons in problem.constraints:
            result = transformation.apply(cons.root_expr, ctx)
            new_expr = insert_tree(problem, relaxed, result.expression, memo)
            relaxed.add_constraint(
                cons.name,
                new_expr,
                cons.lower_bound,
                cons.upper_bound,
            )
            for new_cons in result.constraints:
                new_expr = insert_tree(problem, relaxed, new_cons.root_expr, memo)
                relaxed.add_constraint(
                    new_cons.name,
                    new_expr,
                    new_cons.lower_bound,
                    new_cons.upper_bound,
                )

        print('Original Problem')
        print()
        print_problem(problem)
        print()
        print('Relaxed Problem')
        print()
        print_problem(relaxed)

    def help_message(self):
        return "Save GALINI DAG of the problem as Graphviz Dot file"

    def add_extra_parser_arguments(self, parser):
        parser.add_argument('out', nargs='?')
