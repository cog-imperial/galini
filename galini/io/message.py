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
import datetime

import galini.io.message_pb2 as proto

DEFAULT_LEVEL = 0


def text_message(name, content, level=None, timestamp=None):
    if level is None:
        level = DEFAULT_LEVEL
    msg = message(name, timestamp)
    msg.text.level = level
    msg.text.content = content
    return msg


def tensor_message(name, filename, group, dataset, sizes,
                   timestamp=None):
    msg = message(name, timestamp)
    msg.tensor.filename = filename
    msg.tensor.group_ = group
    msg.tensor.dataset = dataset
    msg.tensor.sizes.extend(sizes)
    return msg


def solve_start_message(name, solver, timestamp=None):
    msg = message(name, timestamp)
    msg.solve_start.solver = solver
    return msg


def solve_end_message(name, solver, timestamp=None):
    msg = message(name, timestamp)
    msg.solve_end.solver = solver
    return msg


def update_variable_message(name, var_name, iteration, value, timestamp=None):
    msg = message(name, timestamp)
    msg.update_variable.name = var_name
    if not isinstance(iteration, list):
        iteration = [iteration]
    msg.update_variable.iteration.extend(iteration)
    msg.update_variable.value = value
    return msg


def add_bab_node_message(name, coordinate, lower_bound, upper_bound,
                         branching_variables=None, timestamp=None):
    msg = message(name, timestamp)
    msg.add_bab_node.coordinate.extend(coordinate)
    msg.add_bab_node.lower_bound = lower_bound
    msg.add_bab_node.upper_bound = upper_bound
    if branching_variables is None:
        branching_variables = []
    for (br_name, br_lower_bound, br_upper_bound) in branching_variables:
        variable_info = msg.add_bab_node.variables_information.add()
        variable_info.variable_name = br_name
        variable_info.lower_bound = br_lower_bound
        variable_info.upper_bound = br_upper_bound
    return msg


def prune_bab_node_message(name, coordinate, timestamp=None):
    msg = message(name, timestamp)
    for value in coordinate:
        msg.prune_bab_node.coordinate.append(value)
    return msg


def message(name, timestamp=None):
    if timestamp is None:
        timestamp = datetime.datetime.utcnow()

    msg = proto.Message()
    if name:
        msg.name = name
    msg.timestamp = timestamp_to_uint64(timestamp)
    return msg


def timestamp_to_uint64(timestamp):
    """Convert timestamp to milliseconds since epoch."""
    return int(timestamp.timestamp() * 1e3)
