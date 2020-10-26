#  Copyright 2020 Francesco Ceccon
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

"""Branch & Cut relaxation interface."""
from galini.relaxations.relax import relax, relax_inequality, update_relaxation_data


class Relaxation:
    def relax(self, model, data):
        raise NotImplementedError('Relaxation.relax')

    def relax_inequality(self, model, ineq_expr, relaxation_side, data):
        raise NotImplementedError('Relaxation.relax')


class DefaultRelaxation(Relaxation):
    def __init__(self, algorithm):
        pass

    def relax(self, model, data):
        return relax(model, data)

    def relax_inequality(self, model, ineq_expr, relaxation_side, data):
        relaxed_ineq = relax_inequality(model, ineq_expr, relaxation_side, data)
        update_relaxation_data(model, data)
        return relaxed_ineq
