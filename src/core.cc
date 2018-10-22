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

#include <pybind11/pybind11.h>

#include "ad/module.h"
#include "expression/module.h"
#include "problem/module.h"

namespace py = pybind11;

namespace galini {

PYBIND11_MODULE(galini_core, m) {
  ad::init_module(m);
  expression::init_module(m);
  problem::init_module(m);
}

} // namespace galini
