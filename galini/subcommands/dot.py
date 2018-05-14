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
"""GALINI dot subcommand."""
from argparse import Namespace, ArgumentParser
from galini.subcommands import CliCommand
from galini.pyomo import read_pyomo_model, dag_from_pyomo_model
from galini.dot import dag_to_pydot_graph


class DotCommand(CliCommand):
    """Command to output Graphiviz dot file of the problem."""
    def execute(self, args: Namespace) -> None:
        assert args.problem
        pyomo_model = read_pyomo_model(args.problem)
        dag = dag_from_pyomo_model(pyomo_model)
        graph = dag_to_pydot_graph(dag)

        if args.out:
            graph.write(args.out)
        else:
            print(graph.to_string())

    def help_message(self) -> str:
        return "Save GALINI DAG of the problem as Graphviz Dot file"

    def add_parser_arguments(self, parser: ArgumentParser) -> None:
        parser.add_argument('problem')
        parser.add_argument('out', nargs='?')
