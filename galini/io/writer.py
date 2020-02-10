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
"""Write messages to binary file."""
from google.protobuf.internal.encoder import _VarintBytes


class MessageWriter(object):
    def __init__(self, writer):
        self._writer = writer

    def write(self, message):
        serialized = message.SerializeToString()
        size = len(serialized)
        self._writer.write(_VarintBytes(size))
        self._writer.write(serialized)
        self._writer.flush()
