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
"""Time and run time related utilities."""
import datetime
import signal


DEFAULT_TIMELIMIT = 86400

class Timelimit(object):
    def __init__(self, timelimit):
        """Timelimit for the run.

        Parameters
        ----------
        timelimit : int
            timelimit in seconds
        start : DateTime
            start time"""
        self.timelimit = timelimit
        self.start = None

    def start_now(self):
        self.start = datetime.datetime.utcnow()

    def seconds_left(self):
        now = datetime.datetime.utcnow()
        difference = now - self.start
        time_left = int(self.timelimit - difference.total_seconds())
        return max(0, time_left)


_timelimit = Timelimit(DEFAULT_TIMELIMIT)
seconds_left = lambda: _timelimit.seconds_left()


class timeout(object):
    def __init__(self, timelimit):
        global _timelimit
        _timelimit.timelimit = timelimit

    def __enter__(self):
        global _timelimit
        _timelimit.start_now()
        signal.signal(signal.SIGALRM, self.handle_timeout)
        signal.alarm(_timelimit.seconds_left())

    def __exit__(self, type, value, traceback):
        signal.alarm(0)

    def handle_timeout(self, signum, frame):
        raise TimeoutError('Timelimit reached.')
