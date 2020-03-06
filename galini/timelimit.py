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

    def set_timelimit(self, t):
        self.timelimit = t

    def elapsed_time(self):
        if self.start is None:
            raise RuntimeError('elapsed_time requires started time')
        now = datetime.datetime.utcnow()
        difference = now - self.start
        return difference.total_seconds()

    def seconds_left(self):
        if self.start is None:
            return DEFAULT_TIMELIMIT
        now = datetime.datetime.utcnow()
        difference = now - self.start
        time_left = int(self.timelimit - difference.total_seconds())
        return max(0, time_left)

    def timeout(self):
        return self.seconds_left() <= 0


def current_time():
    return datetime.datetime.utcnow()


def seconds_elapsed_since(time):
    now = current_time()
    diff = now - time
    return diff.total_seconds()


class timeout(object):
    def __init__(self, seconds, message):
        self.seconds = seconds
        self.message = message

    def handle_timeout(self, signum, frame):
        raise TimeoutError(self.message)

    def __enter__(self):
        signal.signal(signal.SIGALRM, self.handle_timeout)
        signal.alarm(int(self.seconds))

    def __exit__(self, type, value, traceback):
        signal.alarm(0)
