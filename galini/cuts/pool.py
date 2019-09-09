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


class CutNodeStorage:
    def __init__(self, parent_node_storage, pool):
        self.parent_node_storage = parent_node_storage
        if pool is None:
            raise RuntimeError(
                'Trying to create CutNodeStorage without a pool'
            )
        self.pool = pool
        self._cuts_indexes = []

    def add_cut(self, cut):
        self._cuts_indexes.append(cut.index)

    @property
    def cuts(self):
        if self.parent_node_storage is not None:
            for cut in self.parent_node_storage.cuts:
                yield cut
        for cut_index in self._cuts_indexes:
            yield self.pool.cut_at_index(cut_index)


class CutPool:
    def __init__(self, problem):
        self.problem = problem
        self._cuts = []

    def add_cut(self, cut):
        index = len(self._cuts)
        cut.index = index
        self._cuts.append(cut)
        return index

    def cut_at_index(self, index):
        return self._cuts[index]
