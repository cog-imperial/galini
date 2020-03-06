# Copyright 2019 Francesco Ceccon
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

"""Base classes for algorithms."""
import abc


class Algorithm(metaclass=abc.ABCMeta):
    """Base class for all algorithms."""

    name = None

    def __init__(self, galini):
        self.galini = galini

    @property
    def config(self):
        return self.galini.get_configuration_group(self.name)

    def solve(self, model, **kwargs):
        """Solve the optimization problem.

        Arguments
        ---------
        model: ConcreteModel
            the optimization problem
        kwargs: dict
            additional (possibly solver specific) keyword arguments

        Returns
        -------
        Solution
        """
        self.galini.logger.log_solve_start(self.name)
        try:
            solution = self.actual_solve(model, **kwargs)
            self.galini.logger.log_solve_end(self.name)
        except Exception as ex:
            self.galini.logger.log_solve_end(self.name)
            raise ex
        return solution

    @abc.abstractmethod
    def actual_solve(self, model, **kwargs):
        pass
