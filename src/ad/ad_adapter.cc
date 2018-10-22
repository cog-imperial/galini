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

#include "ad_adapter.h"

namespace galini {

namespace ad {

namespace detail {
  // TODO(fra): use pybind11 provided methods when they update
  // release on pypi.
  bool rich_compare(const py::object& self, const py::object& other, int value) {
    int result = PyObject_RichCompareBool(self.ptr(), other.ptr(), value);
    if (result == -1) {
      throw py::error_already_set();
    }
    return result == 1;
  }

}

#define AD_PYOBJECT_ADAPTER_ASSIGN_OPERATOR_IMPL(op, pyop)	\
  void ADPyobjectAdapter::operator op(const ADPyobjectAdapter& other)	\
  { inner_ = inner_.attr("pyop")(other.inner_); }


#define AD_PYOBJECT_ADAPTER_BINARY_OPERATOR_IMPL(op, pyop) \
  ADPyobjectAdapter ADPyobjectAdapter::operator op(const ADPyobjectAdapter& other) const { \
  auto result = inner_.attr("pyop")(other.inner_); \
  return ADPyobjectAdapter(result); }


#define AD_PYOBJECT_ADAPTER_COMPARE_OPERATOR_IMPL(op, pyop) \
  bool ADPyobjectAdapter::operator op(const ADPyobjectAdapter& other) const \
  { return detail::rich_compare(inner_, other.inner_, pyop); }

  AD_PYOBJECT_ADAPTER_ASSIGN_OPERATOR_IMPL(+=, __add__)
  AD_PYOBJECT_ADAPTER_ASSIGN_OPERATOR_IMPL(-=, __sub__)
  AD_PYOBJECT_ADAPTER_ASSIGN_OPERATOR_IMPL(*=, __mul__)
  AD_PYOBJECT_ADAPTER_ASSIGN_OPERATOR_IMPL(/=, __div__)

  AD_PYOBJECT_ADAPTER_BINARY_OPERATOR_IMPL(+, __add__)
  AD_PYOBJECT_ADAPTER_BINARY_OPERATOR_IMPL(-, __sub__)
  AD_PYOBJECT_ADAPTER_BINARY_OPERATOR_IMPL(*, __mul__)
  AD_PYOBJECT_ADAPTER_BINARY_OPERATOR_IMPL(/, __div__)

  AD_PYOBJECT_ADAPTER_COMPARE_OPERATOR_IMPL(==, Py_EQ);
  AD_PYOBJECT_ADAPTER_COMPARE_OPERATOR_IMPL(!=, Py_NE);
  AD_PYOBJECT_ADAPTER_COMPARE_OPERATOR_IMPL(<, Py_LT);
  AD_PYOBJECT_ADAPTER_COMPARE_OPERATOR_IMPL(<=, Py_LE);
  AD_PYOBJECT_ADAPTER_COMPARE_OPERATOR_IMPL(>, Py_GT);
  AD_PYOBJECT_ADAPTER_COMPARE_OPERATOR_IMPL(>=, Py_GE);

#undef AD_PYOBJECT_ADAPTER_ASSIGN_OPERATOR_IMPL
#undef AD_PYOBJECT_ADAPTER_BINARY_OPERATOR_IMPL
#undef AD_PYOBJECT_ADAPTER_COMPARE_OPERATOR_IMPL
}

}
