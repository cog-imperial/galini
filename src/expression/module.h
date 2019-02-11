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
#pragma once

#include <pybind11/pybind11.h>

#include "expression_base.h"
#include "unary_expression.h"
#include "binary_expression.h"
#include "nary_expression.h"

#include "unary_function_expression.h"

#include "variable.h"
#include "auxiliary_variable.h"
#include "constant.h"

namespace galini {

namespace expression {

void init_module(pybind11::module& m);

} // namespace expression

} // namespace galini
