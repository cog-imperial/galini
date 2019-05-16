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
from galini.abb import AlphaBBRelaxation
from galini.dot import dag_to_pydot_graph


class AbbCommand(CliCommandWithProblem):
    def execute_with_problem(self, _model, problem, args):
        abb = AlphaBBRelaxation()
        relaxed_problem = abb.relax(problem)
        graph = dag_to_pydot_graph(relaxed_problem)

        if args.out:
            graph.write(args.out)
        else:
            print(graph.to_string())


    def help_message(self):
        return "Save GALINI DAG of the problem as Graphviz Dot file"

    def add_extra_parser_arguments(self, parser):
        parser.add_argument('out', nargs='?')
