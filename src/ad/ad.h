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
#include "ad_adapter.h"
#include <cppad/cppad.hpp>

namespace galini {

namespace ad {

template<class T>
using AD = CppAD::AD<T>;

using ADFloat = AD<double>;
using ADObject = AD<ADPyobjectAdapter>;


template<class U>
AD<U> pow(const AD<U>& x, const AD<U>& y) { return CppAD::pow(x, y); }

#define FORWARD_UNARY_FUNCTION(name) \
  template<class U> AD<U> name(const AD<U>& x) { return CppAD::name(x); }

FORWARD_UNARY_FUNCTION(abs)
FORWARD_UNARY_FUNCTION(sqrt)
FORWARD_UNARY_FUNCTION(exp)
FORWARD_UNARY_FUNCTION(log)
FORWARD_UNARY_FUNCTION(sin)
FORWARD_UNARY_FUNCTION(cos)
FORWARD_UNARY_FUNCTION(tan)
FORWARD_UNARY_FUNCTION(asin)
FORWARD_UNARY_FUNCTION(acos)
FORWARD_UNARY_FUNCTION(atan)

#undef FORWARD_UNARY_FUNCTION

} // namespace ad

} // namespace galini
