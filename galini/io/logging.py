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

"""GALINI IO logging."""

import logging
from pathlib import Path

import h5py
import numpy as np

from galini.io.message import (
    text_message,
    tensor_message,
    solve_start_message,
    solve_end_message,
    update_variable_message,
    add_bab_node_message,
    prune_bab_node_message,
)
from galini.io.writer import MessageWriter

CRITICAL = logging.CRITICAL
ERROR = logging.ERROR
WARNING = logging.WARNING
INFO = logging.INFO
DEBUG = logging.DEBUG
NOTSET = logging.NOTSET


class LogManager(object):
    """LogManager class for rich log messages.

    If `directory` is `None`, then rich logging will be disabled.
    This object keeps referenecs to the Python logger and output
    files, but does not provide any method to write to them.
    Instantiate a child logger for each solver/run instead.

    Parameters
    ----------
    config : dict-like
        logging configuration
    """
    def __init__(self, config=None):
        self.config = config
        self.has_rich_logging = False
        self._loggers = {}
        self.apply_config(config)

    def apply_config(self, config):
        """Apply config to logger."""
        if config is None:
            config = {}

        level_name = config.get('level', 'INFO')
        if isinstance(level_name, str):
            level_name = logging.getLevelName(level_name)
        self._update_log_level(level_name)

        # delegate some logs to python logging module
        self._pylogger = logging.Logger(__name__)
        self._pylogger.setLevel(self.level)
        if config.get('stdout', False):
            stream_handler = logging.StreamHandler()
            self._pylogger.addHandler(stream_handler)

        if config.get('file') is not None:
            file_handler = logging.FileHandler(config['file'])
            self._pylogger.addHandler(file_handler)

        self._setup_message_log(config)

    def file_path(self, filename):
        """Full path for filename inside logger output dir.

        Parameters
        ----------
        filename : string
            file name

        Returns
        -------
        path or None
            Returns None if rich logging is disabled
        """
        if not self.has_rich_logging:
            return None
        path = self.directory / filename
        return str(path)

    def get_logger(self, name):
        if name in self._loggers:
            return self._loggers[name]
        else:
            logger = Logger(name, manager=self, level=self.level)
            self._loggers[name] = logger
            return logger

    def _update_log_level(self, level):
        self.level = level
        for logger in self._loggers.values():
            logger.level = level

    def _setup_message_log(self, config):
        directory = config.get('directory', None)
        if not directory:
            self.has_rich_logging = False
            return
        self.has_rich_logging = True
        self.directory = Path(directory)
        if not self.directory.exists():
            self.directory.mkdir(exist_ok=True)
        self.messages_file = open(self.directory / 'messages.bin', 'wb')
        self.writer = MessageWriter(self.messages_file)
        self.data_filename = 'data.hdf5'
        self.data_filepath = str(self.directory / self.data_filename)
        # Avoid exception about already open file when
        # re-applying config
        if getattr(self, 'data_file', None):
            self.data_file.close()
        self.data_file = h5py.File(self.data_filepath, 'w')

    def _log_message(self, message):
        if not self.has_rich_logging:
            return
        self.writer.write(message)

    def _log(self, name, lvl, msg, *args, **kwargs):
        if lvl < self.level:
            return
        fmt_msg = msg.format(*args, **kwargs)
        # scrip newline because it's added by pylogger
        if fmt_msg[-1] == '\n':
            pylog_fmt_msg = fmt_msg[:-1]
        else:
            pylog_fmt_msg = fmt_msg
        self._pylogger.log(
            lvl,
            '[{}] {}'.format(name, pylog_fmt_msg),
        )

        message = text_message(name, fmt_msg, level=lvl)
        self._log_message(message)

    def _tensor(self, name, group, dataset, data):
        if not self.has_rich_logging:
            return
        group = '{}/{}'.format(name, group)
        if group is None:
            h5_group = self.data_file
        else:
            if group not in self.data_file:
                self.data_file.create_group(group)
            h5_group = self.data_file[group]
        if dataset not in h5_group:
            data = np.array(data, dtype=np.float)
            h5_group.create_dataset(dataset, data=data)
            message = tensor_message(
                name,
                filename=self.data_filepath,
                group=group,
                dataset=dataset,
                sizes=np.shape(data),
            )
            self._log_message(message)

    def __del__(self):
        if self.has_rich_logging:
            try:
                self.messages_file.close()
                self.data_file.close()
            except:
                pass


class Logger(object):
    def __init__(self, name, manager, level=None):
        self.name = name
        self.manager = manager
        if level is None:
            level = INFO
        self.level = level

    def is_debug(self):
        return self.level <= DEBUG

    def log_message(self, message):
        """Log message to disk."""
        self.manager._log_message(message)

    def debug(self, msg, *args, **kwargs):
        """Log msg with DEBUG level."""
        return self.log(DEBUG, msg, *args, **kwargs)

    def info(self, msg, *args, **kwargs):
        """Log msg with INFO level."""
        return self.log(INFO, msg, *args, **kwargs)

    def warning(self, msg, *args, **kwargs):
        """Log msg with WARNING level."""
        return self.log(WARNING, msg, *args, **kwargs)

    def error(self, msg, *args, **kwargs):
        """Log msg with ERROR level."""
        return self.log(ERROR, msg, *args, **kwargs)

    def log(self, lvl, msg, *args, **kwargs):
        """Log msg with lvl level and unique run id.

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
        if lvl >= self.level:
            self.manager._log(self.name, lvl, msg, *args, **kwargs)

    def log_solve_start(self, solver):
        self.log_message(solve_start_message(
            name=self.name,
            solver=solver,
        ))

    def log_solve_end(self, solver):
        self.log_message(solve_end_message(
            name=self.name,
            solver=solver,
        ))

    def log_add_bab_node(self, coordinate, lower_bound, upper_bound,
                         branching_variables=None):
        self.log_message(add_bab_node_message(
            name=self.name,
            coordinate=coordinate,
            lower_bound=lower_bound,
            upper_bound=upper_bound,
            branching_variables=branching_variables,
        ))

    def log_prune_bab_node(self, coordinate):
        self.log_message(prune_bab_node_message(
            name=self.name,
            coordinate=coordinate,
        ))

    def update_variable(self, var_name, iteration, value):
        self.log_message(update_variable_message(
            name=self.name,
            var_name=var_name,
            iteration=iteration,
            value=value,
        ))

    def tensor(self, group, dataset, data):
        """Log tensor data to data file, if configured.

        Arguments
        ---------
        group : string
            dataset group
        dataset : string
            dataset name
        data : array-like
            the data to log
        """
        return self.manager._tensor(self.name, group, dataset, data)
