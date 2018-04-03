# Copyright 2017 Francesco Ceccon
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

import abc
from galini.math import (
    mpf,
    almosteq,
    almostlte,
)


class FloatHasher(abc.ABC):
    @abc.abstractmethod
    def hash(self, f):
        raise NotImplementedError('hash')


class BTreeFloatHasher(FloatHasher):
    """A floating point hasher that keeps all seen floating
    point numbers ina binary tree.

    Good if the unique values of the floating point numbers in
    the problem are relatively few.
    """

    class Node(object):
        def __init__(self, num, hash_, left=None, right=None):
            self.num = num
            self.hash = hash_
            self.left = left
            self.right = right

    def __init__(self):
        self.root = None
        self.node_count = 0

    def hash(self, f):
        f = mpf(f)
        if self.root is None:
            self.root = self._make_node(f)
            return self.root.hash

        curr_node = self.root
        while True:
            if almosteq(f, curr_node.num):
                return curr_node.hash
            elif almostlte(f, curr_node.num):
                if curr_node.left is None:
                    new_node = self._make_node(f)
                    curr_node.left = new_node
                    return new_node.hash
                else:
                    curr_node = curr_node.left
            else:
                if curr_node.right is None:
                    new_node = self._make_node(f)
                    curr_node.right = new_node
                    return new_node.hash
                else:
                    curr_node = curr_node.right

    def _make_node(self, f):
        node = self.Node(f, self.node_count, None, None)
        self.node_count += 1
        return node


class RoundFloatHasher(FloatHasher):
    """A float hasher that hashes floats up to the n-th
    decimal place.
    """
    def __init__(self, n=2):
        self.n = 10**n

    def hash(self, f):
        return hash(int(f * self.n))
