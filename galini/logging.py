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
from typing import Any, Optional, Union
from pathlib import Path
import logging
import numpy as np

CRITICAL = logging.CRITICAL
ERROR = logging.ERROR
WARNING = logging.WARNING
INFO = logging.INFO
DEBUG = logging.DEBUG
NOTSET = logging.NOTSET


class MessageStorage(object):
    def log(self, lvl: int, msg: str, *args: Any, **kwargs: Any) -> None:
        """Log msg with lvl level.

        Arguments
        ---------
        lvl: int
            logging level
        msg: str
            format string
        args: Any
            arguments passed to msg.format
        kwargs: Any
            keyword arguments passed to msg.format
        """


class MatrixStorage(object):
    pass


class Logger(object):
    """Object used to log data to disk."""
    def __init__(self, message_storage: Optional[MessageStorage] = None,
                 matrix_storage: Optional[MatrixStorage] = None):
        self._message_storage = message_storage
        self._matrix_storage = matrix_storage

    def debug(self, msg: str, *args: Any, **kwargs: Any) -> None:
        """Log msg with DEBUG level."""
        return self.log(DEBUG, msg, *args, **kwargs)

    def info(self, msg: str, *args: Any, **kwargs: Any) -> None:
        """Log msg with INFO level."""
        return self.log(INFO, msg, *args, **kwargs)

    def warning(self, msg: str, *args: Any, **kwargs: Any) -> None:
        """Log msg with WARNING level."""
        return self.log(WARNING, msg, *args, **kwargs)

    def error(self, msg: str, *args: Any, **kwargs: Any) -> None:
        """Log msg with ERROR level."""
        return self.log(ERROR, msg, *args, **kwargs)

    def log(self, lvl: int, msg: str, *args: Any, **kwargs: Any) -> None:
        """Log msg with lvl level.

        Arguments
        ---------
        lvl: int
            logging level
        msg: str
            format string
        args: Any
            arguments passed to msg.format
        kwargs: Any
            keyword arguments passed to msg.format
        """
        if self._message_storage:
            self._message_storage.log(lvl, msg, *args, **kwargs)

    def matrix(self, path: str, mat: np.array):
        """Log numpy.array to archive.

        Arguments
        ---------
        path: str
            heriarchical group of matrix
        mat: numpy.array
            the matrix or array to log
        """
        if self._matrix_storage:
            self._matrix_storage.matrix(path, mat)

    def set_matrix_storage(self, matrix_storage: MatrixStorage) -> None:
        """Set the new matrix storage."""
        self._matrix_storage = matrix_storage

    def set_message_storage(self, message_storage: MessageStorage) -> None:
        """Set the new message storage."""
        self._message_storage = message_storage


class BuiltinLoggingMessageStorage(MessageStorage):
    """A MessageStorage that uses the builtin logging module."""
    def log(self, lvl: int, msg: str, *args: Any, **kwargs: Any) -> None:
        logging.log(lvl, msg, *args, **kwargs)


class Hdf5MatrixStorage(MatrixStorage):
    """A MatrixStorage that logs to an hdf5 file."""
    def __init__(self, h5_file):
        import h5py
        self._file = h5py.File(h5_file, 'w')

    def matrix(self, path: str, mat: np.array) -> None:
        if path not in self._file:
            grp = self._file.create_group(path)
            grp.attrs['size'] = 0
        else:
            grp = self._file[path]

        size = grp.attrs['size']
        grp.create_dataset(str(size), data=mat)
        grp.attrs['size'] += 1


# pylint: disable=invalid-name
_logger = Logger(message_storage=BuiltinLoggingMessageStorage())
debug = _logger.debug
info = _logger.info
warning = _logger.warning
error = _logger.error
log = _logger.log
matrix = _logger.matrix
set_message_storage = _logger.set_message_storage
set_matrix_storage = _logger.set_matrix_storage


def apply_config(config: 'GaliniConfig') -> None:
    config = config.get_group('logging')
    _apply_matrix_storage_config(config)


def _apply_matrix_storage_config(config):
    config = config['matrix_storage']
    if config['class']:
        class_ = config['class']
        if class_ == 'Hdf5MatrixStorage':
            set_matrix_storage(Hdf5MatrixStorage(config['hdf5_file']))
