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
"""Log messages and data to disk."""
from galini_io import Logger


# pylint: disable=invalid-name
_logger = Logger()
debug = _logger.debug
info = _logger.info
warning = _logger.warning
error = _logger.error
log = _logger.log
tensor = _logger.tensor


def apply_config(config):
    """Apply config to current logger."""
    config = config.logging
    _logger.apply_config(config)


def get_logger():
    return _logger
