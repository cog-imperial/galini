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

"""OSiL reader module."""
import xml.etree.ElementTree as ET
import gc
from contextlib import contextmanager
from functools import reduce
import operator as op
import pyomo.environ as aml


def read_osil(filename, objective_prefix=None, constraint_prefix=None):
    """Read OSiL-formatted problem into a Pyomo model.

    It's possible to optionally specify a prefix for objectives/constraints
    to avoid name clashes.

    Parameters
    ----------
    filename : str
        the OSiL file path.
    objective_prefix : str
        a prefix to give all objectives.
    constraint_prefix : str
        a prefix to give all constraints.

    Returns
    -------
    ConcreteModel
        a Pyomo concrete model.
    """
    parser = OsilParser(filename, objective_prefix, constraint_prefix)
    return parser.parse()


NS = {'osil': 'os.optimizationservices.org'}
TYPE_TO_DOMAIN = {
    'I': aml.Integers,
    'C': aml.Reals,
    'B': aml.Binary,
}
STR_TO_SENSE = {
    'min': aml.minimize,
    'max': aml.maximize,
}


@contextmanager
def _gc_disabled():
    if gc.isenabled():
        gc.disable()
        yield
        gc.enable()
    else:
        yield


def _tag_name(node):
    prefix = len('{os.optimizationservices.org}')
    return node.tag[prefix:]


def _instance_name(root):
    name = root.find('osil:instanceHeader/osil:name', NS)
    if name is not None:
        return name.text
    return None


def _instance_variables(root):
    variables = root.find('osil:instanceData/osil:variables', NS)
    num_variables = int(variables.attrib['numberOfVariables'])
    vars_ = variables.getchildren()
    assert len(vars_) == num_variables
    for var in vars_:
        attr = var.attrib
        name = attr['name']
        lb = float(attr['lb']) if 'lb' in attr else 0.0
        ub = float(attr['ub']) if 'ub' in attr else None
        type_ = attr.get('type', 'C')
        if type_ == 'S':
            raise RuntimeError('unsupported var type S')
        bounds = (lb, ub)
        domain = TYPE_TO_DOMAIN[type_]
        yield {
            'name': name,
            'bounds': bounds,
            'domain': domain,
        }


def _instance_objectives(root):
    return root.findall('osil:instanceData/osil:objectives/osil:obj', NS)


def _instance_constraints(root):
    return root.findall('osil:instanceData/osil:constraints/osil:con', NS)


def _quadratic_coefficients(root, i):
    qterms = root.findall(
        'osil:instanceData/osil:quadraticCoefficients/osil:qTerm[@idx="{}"]'.format(i),
        NS
    )
    for qterm in qterms:
        attr = qterm.attrib
        idx_one = int(attr['idxOne'])
        idx_two = int(attr['idxTwo'])
        coef = float(attr['coef'])
        yield {
            'idx_one': idx_one,
            'idx_two': idx_two,
            'coef': coef,
        }


def _flatten(xs):
    return [y for x in xs for y in x]


class SparseLinearCoefficientsStorage(object):
    """Represent linear terms as sparse matrix."""
    def __init__(self, root):
        self.root = root
        if self.root is None:
            return
        self.has_col_idx = self.root.find('osil:colIdx', NS) is not None
        self.has_row_idx = self.root.find('osil:rowIdx', NS) is not None
        if self.has_col_idx == self.has_row_idx:
            raise RuntimeError('Invalid sparse linear storage')

        if self.has_row_idx:
            raise RuntimeError('rowIdx not implemented')

        start = [
            self._expand(el)
            for el in self.root.findall('osil:start/osil:el', NS)
        ]
        self.start = _flatten(start)

        col_idx = [
            self._expand(el)
            for el in self.root.findall('osil:colIdx/osil:el', NS)
        ]
        self.col_idx = _flatten(col_idx)

        value = [
            self._expand(el, float)
            for el in self.root.findall('osil:value/osil:el', NS)
        ]
        self.value = _flatten(value)

    def row(self, i):
        """Return variables at row i with their coefficients."""
        if self.root is None:
            return []
        start_idx = self.start[i]
        end_idx = self.start[i+1]
        vars_with_coef = zip(
            self.value[start_idx:end_idx], self.col_idx[start_idx:end_idx]
        )
        return vars_with_coef

    def _expand(self, ele, text_fun=int):
        mult = int(ele.attrib.get('mult', 1))
        incr = int(ele.attrib.get('incr', 0))
        start = text_fun(ele.text)
        res = [start] * mult
        for i in range(1, mult):
            res[i] = res[i-1] + incr
        return res


class OsilParser(object):
    """Parser for OSiL files."""
    def __init__(self, filename, objective_prefix=None, constraint_prefix=None):
        self.tree = ET.parse(filename)
        self.root = self.tree.getroot()
        self.model = aml.ConcreteModel()
        self.objective_prefix = objective_prefix
        self.constraint_prefix = constraint_prefix

        self.indexed_vars = []

    def _v(self, i):
        vname = self.indexed_vars[i]
        return getattr(self.model, vname)

    def _quadratic_terms(self, i):
        return sum(
            c['coef'] * self._v(c['idx_one']) * self._v(c['idx_two'])
            for c in _quadratic_coefficients(self.root, i)
        )

    # pylint: disable=invalid-name
    def _nonlinear_terms(self, i):
        # pylint: disable=too-many-return-statements, too-many-branches
        def _eval(node):
            name = _tag_name(node)
            cs = node.getchildren()

            if name == 'negate':
                assert len(cs) == 1
                return - _eval(cs[0])
            elif name == 'number':
                return float(node.attrib['value'])
            elif name == 'variable':
                c = float(node.attrib.get('coef', 1))
                return c * self._v(int(node.attrib['idx']))
            elif name == 'power':
                assert len(cs) == 2
                b = _eval(cs[0])
                e = _eval(cs[1])
                return b ** e
            elif name == 'square':
                assert len(cs) == 1
                return _eval(cs[0])**2
            elif name == 'sqrt':
                assert len(cs) == 1
                return aml.sqrt(_eval(cs[0]))
            elif name == 'product':
                return reduce(op.mul, [_eval(c) for c in cs])
            elif name == 'divide':
                assert len(cs) == 2
                return _eval(cs[0]) / _eval(cs[1])
            elif name == 'times':
                assert len(cs) == 2
                return _eval(cs[0]) * _eval(cs[1])
            elif name == 'plus':
                assert len(cs) == 2
                return _eval(cs[0]) + _eval(cs[1])
            elif name == 'sum':
                return sum([_eval(c) for c in cs])
            elif name == 'minus':
                assert len(cs) == 2
                return _eval(cs[0]) - _eval(cs[1])
            elif name == 'abs':
                assert len(cs) == 1
                return abs(_eval(cs[0]))
            elif name == 'exp':
                assert len(cs) == 1
                return aml.exp(_eval(cs[0]))
            elif name == 'ln':
                assert len(cs) == 1
                return aml.log(_eval(cs[0]))
            elif name == 'sin':
                assert len(cs) == 1
                return aml.sin(_eval(cs[0]))
            elif name == 'cos':
                assert len(cs) == 1
                return aml.cos(_eval(cs[0]))
            raise RuntimeError('unhandled tag {}'.format(name))
        nl = self.root.find(
            'osil:instanceData/osil:nonlinearExpressions/osil:nl[@idx="{}"]'.format(i),
            NS,
        )
        if nl is None:
            return 0.0
        assert len(nl.getchildren()) == 1
        with _gc_disabled():
            expr = _eval(nl.getchildren()[0])
        return expr

    def _objective_linear(self, objective):
        assert len(objective.getchildren()) == len(objective.findall('osil:coef', NS))
        return sum(
            float(c.text) * self._v(int(c.attrib['idx']))
            for c in objective.getchildren()
        )

    def _objective_name(self, objective):
        if self.objective_prefix is None:
            return objective.attrib['name']
        return self.objective_prefix + objective.attrib['name']

    def _constraint_name(self, constraint):
        if self.constraint_prefix is None:
            return constraint.attrib['name']
        return self.constraint_prefix + constraint.attrib['name']

    # pylint: disable=too-many-locals
    def parse(self):
        """Do the parsing of the file."""
        self.model.name = _instance_name(self.root)

        for var_def in _instance_variables(self.root):
            new_var = aml.Var(
                bounds=var_def['bounds'],
                domain=var_def['domain'],
            )
            setattr(self.model, var_def['name'], new_var)
            self.indexed_vars.append(var_def['name'])

        for i, objective in enumerate(_instance_objectives(self.root)):
            linear = self._objective_linear(objective)
            quad = self._quadratic_terms(-i-1)
            nl = self._nonlinear_terms(-i-1)
            expr = linear + quad + nl
            obj_name = self._objective_name(objective)
            sense = STR_TO_SENSE[objective.attrib['maxOrMin']]
            obj = aml.Objective(sense=sense, expr=expr)
            setattr(self.model, obj_name, obj)

        lcc = self.root.find('osil:instanceData/osil:linearConstraintCoefficients', NS)
        linear_coefficients = SparseLinearCoefficientsStorage(lcc)
        for i, constraint in enumerate(_instance_constraints(self.root)):
            linear = sum(c * self._v(v) for c, v in linear_coefficients.row(i))
            quad = self._quadratic_terms(i)
            nl = self._nonlinear_terms(i)
            expr = linear + quad + nl
            cons_name = self._constraint_name(constraint)
            lb = float(constraint.attrib.get('lb', '-inf'))
            ub = float(constraint.attrib.get('ub', 'inf'))
            cons = aml.Constraint(expr=lb <= expr <= ub)
            setattr(self.model, cons_name, cons)

        return self.model
