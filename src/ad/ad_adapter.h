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

#if defined(CPPAD_CPPAD_HPP)
#error "Include ad_adapter.hpp before cppad/cppad.hpp"
#endif

#include <Python.h>
#include <pybind11/pybind11.h>
#include <cppad/base_require.hpp>

namespace py = pybind11;

namespace galini {

namespace ad {

#define AD_PYOBJECT_ADAPTER_ASSIGN_OPERATOR(op)	\
  void operator op(const ADPyobjectAdapter& other);

#define AD_PYOBJECT_ADAPTER_BINARY_OPERATOR(op) \
  ADPyobjectAdapter operator op(const ADPyobjectAdapter& other) const;

#define AD_PYOBJECT_ADAPTER_COMPARE_OPERATOR(op) \
  bool operator op(const ADPyobjectAdapter& other) const;


/* Adapter to use Python objects as Base type for CppAD AD<Base>.
 *
 * https://coin-or.github.io/CppAD/doc/base_require.htm
 */
class ADPyobjectAdapter {
public:
  explicit ADPyobjectAdapter()
    : ADPyobjectAdapter(0) {}
  explicit ADPyobjectAdapter(int i)
    : ADPyobjectAdapter(py::int_(i)) {}
  explicit ADPyobjectAdapter(py::object inner)
    : inner_(inner) {}

  ADPyobjectAdapter(const ADPyobjectAdapter& other)
    : ADPyobjectAdapter(other.inner_) {}

  ~ADPyobjectAdapter() = default;

  void operator=(const ADPyobjectAdapter& other) {
    inner_ = other.inner_;
  }

  void operator=(py::object other) {
    inner_ = other;
  }

  ADPyobjectAdapter operator+() const {
    return *this;
  }

  ADPyobjectAdapter operator-() const {
    auto neg = inner_.attr("__neg__")();
    return ADPyobjectAdapter(neg);
  }

  AD_PYOBJECT_ADAPTER_ASSIGN_OPERATOR(+=)
  AD_PYOBJECT_ADAPTER_ASSIGN_OPERATOR(-=)
  AD_PYOBJECT_ADAPTER_ASSIGN_OPERATOR(*=)
  AD_PYOBJECT_ADAPTER_ASSIGN_OPERATOR(/=)

  AD_PYOBJECT_ADAPTER_BINARY_OPERATOR(+)
  AD_PYOBJECT_ADAPTER_BINARY_OPERATOR(-)
  AD_PYOBJECT_ADAPTER_BINARY_OPERATOR(*)
  AD_PYOBJECT_ADAPTER_BINARY_OPERATOR(/)

  AD_PYOBJECT_ADAPTER_COMPARE_OPERATOR(==)
  AD_PYOBJECT_ADAPTER_COMPARE_OPERATOR(!=)
  AD_PYOBJECT_ADAPTER_COMPARE_OPERATOR(<)
  AD_PYOBJECT_ADAPTER_COMPARE_OPERATOR(<=)
  AD_PYOBJECT_ADAPTER_COMPARE_OPERATOR(>)
  AD_PYOBJECT_ADAPTER_COMPARE_OPERATOR(>=)

  py::object inner() const {
    return inner_;
  }
private:
  py::object inner_;
};

#undef AD_PYOBJECT_ADAPTER_ASSIGN_OPERATOR
#undef AD_PYOBJECT_ADAPTER_BINARY_OPERATOR
#undef AD_PYOBJECT_ADAPTER_COMPARE_OPERATOR

} // namespace ad

} // namespace galini
