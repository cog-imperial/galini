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
"""Read messages from binary file."""
import time
from google.protobuf.internal.decoder import _DecodeVarint32
import galini.io.message_pb2 as proto


class MessageReader(object):
    def __init__(self, reader, stream=False):
        self._reader = reader
        self._stream = stream

    def __iter__(self):
        while True:
            msg = self.read_next()
            if msg:
                yield msg
            else:
                return

    def read_next(self):
        while True:
            size = self._read_size()
            if size == 0 and not self._stream:
                return
            if size == 0:
                time.sleep(0.01)
                continue

            buf = self._read_exactly(size)
            msg = proto.Message()
            msg.ParseFromString(buf)
            return msg

    def _read_size(self):
        buf = self._reader.read(1)
        if buf == b'':
            return 0

        # Decode varint size, from documentation:
        #
        # Each byte in a varint, except the last byte, has the most
        # significant bit (msb) set – this indicates that there are
        # further bytes to come. The lower 7 bits of each byte are
        # used to store the two's complement representation of the
        # number in groups of 7 bits, least significant group first.
        while (bytearray(buf)[-1] & 0x80) >> 7 == 1:
            new_byte = self._reader.read(1)
            if new_byte == b'':
                raise EOFError('Unexpected EOF when reading size')
            buf += new_byte

        size, _ = _DecodeVarint32(buf, 0)
        return size

    def _read_exactly(self, size):
        data = b''
        remaining = size
        while remaining > 0:
            new_data = self._reader.read(remaining)
            data += new_data
            remaining -= len(new_data)
        return data
