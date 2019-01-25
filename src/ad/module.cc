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
#include "expression_tree_data.h"
#include "func.h"


namespace galini {

namespace ad {

template<class U>
void init_adfunc(py::module& m, const char *name) {
  py::class_<ADFunc<U>>(m, name)
    .def("forward", &ADFunc<U>::forward)
    .def("reverse", &ADFunc<U>::reverse)
    .def("hessian", py::overload_cast<const std::vector<U>&, std::size_t>(&ADFunc<U>::hessian))
    .def("hessian",
	 py::overload_cast<const std::vector<U>&, const std::vector<U>&>(&ADFunc<U>::hessian));
}

void init_module(py::module& m) {
  py::class_<AD<double>>(m, "AD[float]");
  py::class_<AD<py::object>>(m, "AD[object]");

  py::class_<ExpressionTreeData>(m, "ExpressionTreeData")
    .def("vertices", &ExpressionTreeData::vertices)
    .def("eval", py::overload_cast<const std::vector<double>&>(&ExpressionTreeData::eval<double, double>, py::const_))
    .def("eval", py::overload_cast<const std::vector<double>&, const std::vector<index_t>&>(&ExpressionTreeData::eval<double, double>, py::const_))
    .def("eval", (ADFunc<py::object> (ExpressionTreeData::*)(const std::vector<py::object>&) const) &ExpressionTreeData::eval);;

  init_adfunc<double>(m, "ADFunc[float]");
  init_adfunc<py::object>(m, "ADFunc[object]");
}

} // namespace ad

} // namespace galini
