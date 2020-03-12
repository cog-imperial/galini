#  Copyright 2019 Francesco Ceccon
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

"""Problem cut pool."""

import pyomo.environ as pe


class CutNodeStorage:
    def __init__(self, parent_node_storage, pool):
        self.parent_node_storage = parent_node_storage
        if pool is None:
            raise RuntimeError(
                'Trying to create CutNodeStorage without a pool'
            )
        self.pool = pool
        self._cuts = []

    def add_cut(self, inequality):
        new_cut = self.pool.add_cut(inequality)
        self._cuts.append(new_cut)
        return new_cut

    @property
    def cuts(self):
        if self.parent_node_storage is not None:
            for cut in self.parent_node_storage.cuts:
                yield cut
        for cut in self._cuts:
            yield cut


class CutPool:
    def __init__(self, linear_model):
        self._model = linear_model
        self._cut_pool = pe.Block()
        self._cut_pool.cuts = pe.ConstraintList()
        self._model.cut_pool = self._cut_pool

    def add_cut(self, inequality):
        return self._cut_pool.cuts.add(inequality)

    def deactivate_all(self):
        for cut in self._cut_pool.cuts.values():
            cut.deactivate()
