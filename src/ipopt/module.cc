/* Copyright 2018 Francesco Ceccon

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
======================================================================== */
#include <pybind11/stl.h>

#include "module.h"
#include "ipopt_solve.h"

namespace py = pybind11;

namespace galini {

namespace ipopt {

void init_module(py::module& m) {
  m.def("ipopt_solve", &ipopt_solve);

  py::class_<IpoptSolution, std::shared_ptr<IpoptSolution>> solution(m, "IpoptSolution");
  solution
    .def_readonly("status", &IpoptSolution::status)
    .def_readonly("x", &IpoptSolution::x)
    .def_readonly("zl", &IpoptSolution::zl)
    .def_readonly("zu", &IpoptSolution::zu)
    .def_readonly("g", &IpoptSolution::g)
    .def_readonly("lambda", &IpoptSolution::lambda)
    .def_readonly("objective_value", &IpoptSolution::objective_value);

  py::enum_<IpoptSolution::status_type>(solution, "StatusType")
    .value("not_defined", IpoptSolution::status_type::not_defined)
    .value("success", IpoptSolution::status_type::success)
    .value("maxiter_exceeded", IpoptSolution::status_type::maxiter_exceeded)
    .value("stop_at_tiny_step", IpoptSolution::status_type::stop_at_tiny_step)
    .value("stop_at_acceptable_point", IpoptSolution::status_type::stop_at_acceptable_point)
    .value("local_infeasibility", IpoptSolution::status_type::local_infeasibility)
    .value("user_requested_stop", IpoptSolution::status_type::user_requested_stop)
    .value("feasible_point_found", IpoptSolution::status_type::feasible_point_found)
    .value("diverging_iterates", IpoptSolution::status_type::diverging_iterates)
    .value("restoration_failure", IpoptSolution::status_type::restoration_failure)
    .value("error_in_step_computation", IpoptSolution::status_type::error_in_step_computation)
    .value("invalid_number_detected", IpoptSolution::status_type::invalid_number_detected)
    .value("too_few_degrees_of_freedom", IpoptSolution::status_type::too_few_degrees_of_freedom)
    .value("internal_error", IpoptSolution::status_type::internal_error)
    .value("unknown", IpoptSolution::status_type::unknown)
    .export_values();
}

} // namespace ipopt

} // namespace galini
