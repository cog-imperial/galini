#  Copyright 2020 Francesco Ceccon
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.

import pyomo.environ as pyo


VAR_TYPES = [pyo.Reals, pyo.Integers, pyo.Binary]
DEFAULT_VAR_TYPE = {
    'C': 0,
    'I': 1,
    'B': 2,
}


def _is_comment(line):
    if len(line) == 0:
        return True
    return line[0] in ['!', '#', '%']


def _split_line(line, count, parse):
    if count == 1:
        return parse(line.split()[0])
    elements = line.split()[:count]
    return [p(v) for p, v in zip(parse, elements)]


class _QPLibParser:
    def __init__(self):
        self._state = 'name'

        self._name = None
        self._type = None
        self._sense = None

        self._num_variables = None

        # Only if <> **N, **B
        self._num_constraints = None

        # Only if <> L**
        self._obj_q_nnz_count = None
        self._obj_q_nnz_iter = None
        self._obj_q_entries = dict()

        self._b0_default = None

        self._obj_b_nnz_count = None
        self._obj_b_nnz_iter = None
        self._obj_b_entries = dict()

        self._obj_const = None

        # Only if <> **N, **B, **L
        self._con_q_nnz_count = None
        self._con_q_nnz_iter = None
        self._con_q_entries = dict()

        # Only if <> **N, **B
        self._con_b_nnz_count = None
        self._con_b_nnz_iter = None
        self._con_b_entries = dict()

        self._infinity = None

        # Only if <> **N, **B
        self._cl_default = None
        self._cl_nnz_count = None
        self._cl_nnz_iter = None
        self._cl_entries = dict()

        # Only if <> **N, **B
        self._cu_default = None
        self._cu_nnz_count = None
        self._cu_nnz_iter = None
        self._cu_entries = dict()

        # Only if <> *B*
        self._l_default = None
        self._l_nnz_count = None
        self._l_nnz_iter = None
        self._l_entries = dict()

        # Only if <> *B*
        self._u_default = None
        self._u_nnz_count = None
        self._u_nnz_iter = None
        self._u_entries = dict()

        # Only if <> *C*, *B*, *I*
        self._v_default = None
        self._v_nnz_count = None
        self._v_nnz_iter = None
        self._v_entries = dict()

        self._x0_default = None
        self._x0_nnz_count = None
        self._x0_nnz_iter = None
        self._x0_entries = dict()

        # Only if <> **N, **B
        self._y0_default = None
        self._y0_nnz_count = None
        self._y0_nnz_iter = None
        self._y0_entries = dict()

        self._z0_default = None
        self._z0_nnz_count = None
        self._z0_nnz_iter = None
        self._z0_entries = dict()

        self._var_name_nnz_count = None
        self._var_name_nnz_iter = None
        self._var_name_nnz_entries = dict()

        self._con_name_nnz_count = None
        self._con_name_nnz_iter = None
        self._con_name_nnz_entries = dict()

    def process_line(self, line):
        line = line.strip()
        if _is_comment(line):
            return

        processed = False
        while not processed:
            callback = getattr(self, '_process_{}'.format(self._state))
            processed = callback(line)

    def _process_name(self, line):
        self._name = _split_line(line, 1, str)
        self._state = 'type'
        return True

    def _process_type(self, line):
        self._type = _split_line(line, 1, str)
        self._state = 'sense'
        return True

    def _process_sense(self, line):
        line = _split_line(line, 1, str)
        if line.lower() == 'minimize':
            self._sense = pyo.minimize
        else:
            self._sense = pyo.maximize
        self._state = 'num_variables'
        return True

    def _process_num_variables(self, line):
        self._num_variables = _split_line(line, 1, int)
        self._state = 'num_constraints'
        return True

    def _process_num_constraints(self, line):
        if self._type[2] not in ['B', 'N']:
            self._num_constraints = _split_line(line, 1, int)
            self._state = 'obj_q_nnz_count'
            return True
        else:
            self._num_constraints = 0
            self._state = 'obj_q_nnz_count'
            return False

    def _process_obj_q_nnz_count(self, line):
        if self._type[0] in ['L']:
            self._obj_q_nnz_count = 0
            self._state = 'b_default'
            return False
        else:
            self._obj_q_nnz_count = _split_line(line, 1, int)
            self._obj_q_nnz_iter = 0
            self._state = 'obj_q_entries'
            return True

    def _process_obj_q_entries(self, line):
        if self._obj_q_nnz_iter >= self._obj_q_nnz_count:
            self._state = 'b_default'
            return False
        else:
            self._obj_q_nnz_iter += 1
            i, j, q = _split_line(line, 3, [int, int, float])
            self._obj_q_entries[(i, j)] = q
            return True

    def _process_b_default(self, line):
        self._b0_default = _split_line(line, 1, float)
        self._state = 'obj_b_nnz_count'
        return True

    def _process_obj_b_nnz_count(self, line):
        self._obj_b_nnz_count = _split_line(line, 1, int)
        self._obj_b_nnz_iter = 0
        self._state = 'obj_b_entries'
        return True

    def _process_obj_b_entries(self, line):
        if self._obj_b_nnz_iter >= self._obj_b_nnz_count:
            self._state = 'obj_const'
            return False
        else:
            self._obj_b_nnz_iter += 1
            j, b = _split_line(line, 2, [int, float])
            self._obj_b_entries[j] = b
            return True

    def _process_obj_const(self, line):
        self._obj_const = _split_line(line, 1, float)
        self._state = 'con_q_nnz_count'
        return True

    def _process_con_q_nnz_count(self, line):
        if self._type[2] in ['N', 'B', 'L']:
            self._state = 'con_b_nnz_count'
            return False
        else:
            self._con_q_nnz_count = _split_line(line, 1, int)
            self._con_q_nnz_iter = 0
            self._state = 'con_q_entries'
            return True

    def _process_con_q_entries(self, line):
        if self._con_q_nnz_iter >= self._con_q_nnz_count:
            self._state = 'con_b_nnz_count'
            return False
        else:
            self._con_q_nnz_iter += 1
            i, h, k, q = _split_line(line, 4, [int, int, int, float])
            if i not in self._con_q_entries:
                self._con_q_entries[i] = dict()
            self._con_q_entries[i][(h, k)] = q
            return True

    def _process_con_b_nnz_count(self, line):
        if self._type[2] in ['N', 'B']:
            self._state = 'infinity'
            return False
        else:
            self._con_b_nnz_count = _split_line(line, 1, int)
            self._con_b_nnz_iter = 0
            self._state = 'con_b_entries'
            return True

    def _process_con_b_entries(self, line):
        if self._con_b_nnz_iter >= self._con_b_nnz_count:
            self._state = 'infinity'
            return False
        else:
            self._con_b_nnz_iter += 1
            i, j, b = _split_line(line, 3, [int, int, float])
            if i not in self._con_b_entries:
                self._con_b_entries[i] = dict()
            self._con_b_entries[i][j] = b
            return True

    def _process_infinity(self, line):
        self._infinity = _split_line(line, 1, float)
        self._state = 'cl_default'
        return True

    def _process_cl_default(self, line):
        if self._type[2] in ['B', 'N']:
            self._state = 'l_default'
            return False
        else:
            self._cl_default = _split_line(line, 1, float)
            self._state = 'cl_nnz_count'
            return True

    def _process_cl_nnz_count(self, line):
        self._cl_nnz_count = _split_line(line, 1, int)
        self._cl_nnz_iter = 0
        self._state = 'cl_entries'
        return True

    def _process_cl_entries(self, line):
        if self._cl_nnz_iter >= self._cl_nnz_count:
            self._state = 'cu_default'
            return False
        else:
            self._cl_nnz_iter += 1
            i, c = _split_line(line, 2, [int, float])
            self._cl_entries[i] = c
            return True

    def _process_cu_default(self, line):
        self._cu_default = _split_line(line, 1, float)
        self._state = 'cu_nnz_count'
        return True

    def _process_cu_nnz_count(self, line):
        self._cu_nnz_count = _split_line(line, 1, int)
        self._cu_nnz_iter = 0
        self._state = 'cu_entries'
        return True

    def _process_cu_entries(self, line):
        if self._cu_nnz_iter >= self._cu_nnz_count:
            self._state = 'l_default'
            return False
        else:
            self._cu_nnz_iter += 1
            i, c = _split_line(line, 2, [int, float])
            self._cu_entries[i] = c
            return True

    def _process_l_default(self, line):
        if self._type[1] in ['B']:
            self._state = 'u_default'
            self._l_default = 0
            return False
        else:
            self._l_default = _split_line(line, 1, float)
            self._state = 'l_nnz_count'
            return True

    def _process_l_nnz_count(self, line):
        self._l_nnz_count = _split_line(line, 1, int)
        self._l_nnz_iter = 0
        self._state = 'l_entries'
        return True

    def _process_l_entries(self, line):
        if self._l_nnz_iter >= self._l_nnz_count:
            self._state = 'u_default'
            return False
        else:
            self._l_nnz_iter += 1
            i, c = _split_line(line, 2, [int, float])
            self._l_entries[i] = c
            return True

    def _process_u_default(self, line):
        if self._type[1] in ['B']:
            self._state = 'v_default'
            self._u_default = 1
            return False
        else:
            self._u_default = _split_line(line, 1, float)
            self._state = 'u_nnz_count'
            return True

    def _process_u_nnz_count(self, line):
        self._u_nnz_count = _split_line(line, 1, int)
        self._u_nnz_iter = 0
        self._state = 'u_entries'
        return True

    def _process_u_entries(self, line):
        if self._u_nnz_iter >= self._u_nnz_count:
            self._state = 'v_default'
            return False
        else:
            self._u_nnz_iter += 1
            i, c = _split_line(line, 2, [int, float])
            self._u_entries[i] = c
            return True

    def _process_v_default(self, line):
        if self._type[1] in ['C', 'B', 'I']:
            self._state = 'x0_default'
            self._v_default = DEFAULT_VAR_TYPE[self._type[1]]
            return False
        self._v_default = _split_line(line, 1, int)
        self._state = 'v_nnz_count'
        return True

    def _process_v_nnz_count(self, line):
        if self._type[1] in ['C', 'B', 'I']:
            self._state = 'x0_nnz_count'
            return False
        else:
            self._v_nnz_count = _split_line(line, 1, int)
            self._v_nnz_iter = 0
            self._state = 'v_entries'
            return True

    def _process_v_entries(self, line):
        if self._v_nnz_iter >= self._v_nnz_count:
            self._state = 'x0_default'
            return False
        else:
            self._v_nnz_iter += 1
            i, c = _split_line(line, 2, [int, int])
            self._v_entries[i] = c
            return True

    def _process_x0_default(self, line):
        self._x0_default = _split_line(line, 1, float)
        self._state = 'x0_nnz_count'
        return True

    def _process_x0_nnz_count(self, line):
        self._x0_nnz_count = _split_line(line, 1, int)
        self._x0_nnz_iter = 0
        self._state = 'x0_entries'
        return True

    def _process_x0_entries(self, line):
        if self._x0_nnz_iter >= self._x0_nnz_count:
            self._state = 'y0_default'
            return False
        else:
            self._x0_nnz_iter += 1
            i, c = _split_line(line, 2, [int, float])
            self._x0_entries[i] = c
            return True

    def _process_y0_default(self, line):
        if self._type[2] in ['B', 'N']:
            self._state = 'z0_default'
            return False
        else:
            self._y0_default = _split_line(line, 1, float)
            self._state = 'y0_nnz_count'
            return True

    def _process_y0_nnz_count(self, line):
        self._y0_nnz_count = _split_line(line, 1, int)
        self._y0_nnz_iter = 0
        self._state = 'y0_entries'
        return True

    def _process_y0_entries(self, line):
        if self._y0_nnz_iter >= self._y0_nnz_count:
            self._state = 'z0_default'
            return False
        else:
            self._y0_nnz_iter += 1
            i, c = _split_line(line, 2, [int, float])
            self._y0_entries[i] = c
            return True

    def _process_z0_default(self, line):
        self._z0_default = _split_line(line, 1, float)
        self._state = 'z0_nnz_count'
        return True

    def _process_z0_nnz_count(self, line):
        self._z0_nnz_count = _split_line(line, 1, int)
        self._z0_nnz_iter = 0
        self._state = 'z0_entries'
        return True

    def _process_z0_entries(self, line):
        if self._z0_nnz_iter >= self._y0_nnz_count:
            self._state = 'var_name_nnz_count'
            return False
        else:
            self._z0_nnz_iter += 1
            i, c = _split_line(line, 2, [int, float])
            self._z0_entries[i] = c
            return True

    def _process_var_name_nnz_count(self, line):
        self._var_name_nnz_count = _split_line(line, 1, int)
        self._var_name_nnz_iter = 0
        self._state = 'var_name_entries'
        return True

    def _process_var_name_entries(self, line):
        if self._var_name_nnz_iter >= self._var_name_nnz_count:
            self._state = 'con_name_nnz_count'
            return False
        else:
            self._var_name_nnz_iter += 1
            i, n = _split_line(line, 2, [int, str])
            self._var_name_nnz_entries[i] = n
            return True

    def _process_con_name_nnz_count(self, line):
        self._con_name_nnz_count = _split_line(line, 1, int)
        self._con_name_nnz_iter = 0
        self._state = 'con_name_entries'
        return True

    def _process_con_name_entries(self, line):
        if self._con_name_nnz_iter >= self._con_name_nnz_count:
            self._state = 'done'
            return False
        else:
            self._con_name_nnz_iter += 1
            i, n = _split_line(line, 2, [int, str])
            self._con_name_nnz_entries[i] = n
            return True

    def _process_done(self, line):
        raise RuntimeError('Finished parsing file but read line: {}'.format(line))

    @property
    def model(self):
        model = pyo.ConcreteModel(name=self._name)

        # build variables
        default_var_lb = self._l_default
        default_var_ub = self._u_default
        default_var_x0 = self._x0_default
        variables = []
        for i in range(self._num_variables):
            idx = i + 1
            var_type = VAR_TYPES[self._v_entries.get(idx, self._v_default)]
            var_name = self._var_name_nnz_entries.get(idx, 'x{}'.format(idx))
            var_lb = self._l_entries.get(idx, default_var_lb)
            var_ub = self._u_entries.get(idx, default_var_ub)
            var_x0 = self._x0_entries.get(idx, default_var_x0)
            if var_type in [pyo.Integers, pyo.Binary]:
                var_x0 = int(var_x0)
            var = pyo.Var(
                name=var_name,
                domain=var_type,
                bounds=(var_lb, var_ub),
                initialize=lambda _: var_x0
            )
            variables.append(var)
            setattr(model, var_name, var)

        # build objective
        obj_quad = self._obj_const
        for (i, j), q in self._obj_q_entries.items():
            # d = 2.0 if i == j else 1.0
            d = 2.0
            obj_quad += (q / d) * variables[i-1] * variables[j-1]

        obj_linear = pyo.quicksum(
            [v * self._obj_b_entries.get(i + 1, self._b0_default) for i, v in enumerate(variables)],
            linear=False
        )
        model.obj = pyo.Objective(expr=obj_quad + obj_linear, sense=self._sense)

        # build constraints
        for i in range(self._num_constraints):
            con_idx = i + 1
            con_quad = 0.0
            if con_idx not in self._con_q_entries and con_idx not in self._con_b_entries:
                continue
            if con_idx in self._con_q_entries:
                for (h, k), q in self._con_q_entries[con_idx].items():
                    # d = 2.0 if h == k else 1.0
                    d = 2.0
                    con_quad += (q / d) * variables[h-1] * variables[k-1]
            con_linear = 0.0
            if con_idx in self._con_b_entries:
                con_linear = pyo.quicksum(
                    [b * variables[j-1] for j, b in self._con_b_entries[con_idx].items()],
                    linear=False,
                )
            con_name = self._con_name_nnz_entries.get(i+1, 'e{}'.format(i+1))
            con_lb = self._cl_entries.get(i+1, self._cl_default)
            con_ub = self._cu_entries.get(i+1, self._cu_default)
            con_expr = pyo.inequality(con_lb, con_quad + con_linear, con_ub)
            con = pyo.Constraint(name=con_name, expr=con_expr)
            setattr(model, con_name, con)

        return model


def read_qplib(filename, **kwargs):
    parser = _QPLibParser()
    with open(filename) as f:
        for line in f:
            parser.process_line(line)
    return parser.model