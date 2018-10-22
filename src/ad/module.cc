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
#include "module.h"

#include "ad_data.h"

namespace galini {

namespace ad {

void init_module(py::module& m) {
  py::class_<AD<double>>(m, "AD[float]");
  py::class_<AD<ADPyobjectAdapter>>(m, "AD[object]");

  py::class_<ExpressionTreeData>(m, "ExpressionTreeData")
    .def("vertices", &ExpressionTreeData::vertices);
}

} // namespace ad

} // namespace galini
