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
  { inner_ = inner_.attr(#pyop)(other.inner_); }


#define AD_PYOBJECT_ADAPTER_BINARY_OPERATOR_IMPL(op, pyop) \
  ADPyobjectAdapter ADPyobjectAdapter::operator op(const ADPyobjectAdapter& other) const { \
  auto result = inner_.attr(#pyop)(other.inner_); \
  return ADPyobjectAdapter(result); }


#define AD_PYOBJECT_ADAPTER_COMPARE_OPERATOR_IMPL(op, pyop) \
  bool ADPyobjectAdapter::operator op(const ADPyobjectAdapter& other) const \
  { return detail::rich_compare(inner_, other.inner_, pyop); }


#define AD_PYOBJECT_ADAPTER_UNARY_FUNC_IMPL(func) \
  ADPyobjectAdapter ADPyobjectAdapter::func() const	\
  { return ADPyobjectAdapter(inner_.attr(#func)()); }

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

  AD_PYOBJECT_ADAPTER_UNARY_FUNC_IMPL(acos)
  AD_PYOBJECT_ADAPTER_UNARY_FUNC_IMPL(asin)
  AD_PYOBJECT_ADAPTER_UNARY_FUNC_IMPL(atan)
  AD_PYOBJECT_ADAPTER_UNARY_FUNC_IMPL(cos)
  AD_PYOBJECT_ADAPTER_UNARY_FUNC_IMPL(cosh)
  AD_PYOBJECT_ADAPTER_UNARY_FUNC_IMPL(exp)
  AD_PYOBJECT_ADAPTER_UNARY_FUNC_IMPL(fabs)
  AD_PYOBJECT_ADAPTER_UNARY_FUNC_IMPL(log)
  AD_PYOBJECT_ADAPTER_UNARY_FUNC_IMPL(log10)
  AD_PYOBJECT_ADAPTER_UNARY_FUNC_IMPL(sin)
  AD_PYOBJECT_ADAPTER_UNARY_FUNC_IMPL(sinh)
  AD_PYOBJECT_ADAPTER_UNARY_FUNC_IMPL(sqrt)
  AD_PYOBJECT_ADAPTER_UNARY_FUNC_IMPL(tan)
  AD_PYOBJECT_ADAPTER_UNARY_FUNC_IMPL(tanh)
  AD_PYOBJECT_ADAPTER_UNARY_FUNC_IMPL(abs)
  AD_PYOBJECT_ADAPTER_UNARY_FUNC_IMPL(sign)

#undef AD_PYOBJECT_ADAPTER_ASSIGN_OPERATOR_IMPL
#undef AD_PYOBJECT_ADAPTER_BINARY_OPERATOR_IMPL
#undef AD_PYOBJECT_ADAPTER_COMPARE_OPERATOR_IMPL
#undef AD_PYOBJECT_ADAPTER_UNARY_FUNC_IMPL

bool ADPyobjectAdapter::is_zero() const {
  return py::bool_(inner_.attr("is_zero")());
}

bool ADPyobjectAdapter::is_one() const {
  return py::bool_(inner_.attr("is_one")());
}

ADPyobjectAdapter ADPyobjectAdapter::pow(const ADPyobjectAdapter& other) const {
  return ADPyobjectAdapter(inner_.attr("__pow__")(other.inner_));
}

} // namespace ad

} // namespace galini


namespace CppAD {

bool IdenticalCon(const galini::ad::ADPyobjectAdapter& x) {
  return true;
}

bool IdenticalPar(const galini::ad::ADPyobjectAdapter& x) {
  return true;
}

bool IdenticalZero(const galini::ad::ADPyobjectAdapter& x) {
  return x.is_zero();
}

bool IdenticalOne(const galini::ad::ADPyobjectAdapter& x) {
  return x.is_one();
}

bool IdenticalEqualCon(const galini::ad::ADPyobjectAdapter& x,
		       const galini::ad::ADPyobjectAdapter& y) {
  return x == y;
}

bool IdenticalEqualPar(const galini::ad::ADPyobjectAdapter& x,
		       const galini::ad::ADPyobjectAdapter& y) {
  return x == y;
}

galini::ad::ADPyobjectAdapter pow(const galini::ad::ADPyobjectAdapter& x,
				  const galini::ad::ADPyobjectAdapter& y) {
  return x.pow(y);
}

#define AD_PYOBJECT_ADAPTER_UNARY_FUNC_IMPL(func) \
  galini::ad::ADPyobjectAdapter func(const galini::ad::ADPyobjectAdapter& x) \
  { return x.func(); }

AD_PYOBJECT_ADAPTER_UNARY_FUNC_IMPL(acos)
AD_PYOBJECT_ADAPTER_UNARY_FUNC_IMPL(asin)
AD_PYOBJECT_ADAPTER_UNARY_FUNC_IMPL(atan)
AD_PYOBJECT_ADAPTER_UNARY_FUNC_IMPL(cos)
AD_PYOBJECT_ADAPTER_UNARY_FUNC_IMPL(cosh)
AD_PYOBJECT_ADAPTER_UNARY_FUNC_IMPL(exp)
AD_PYOBJECT_ADAPTER_UNARY_FUNC_IMPL(fabs)
AD_PYOBJECT_ADAPTER_UNARY_FUNC_IMPL(log)
AD_PYOBJECT_ADAPTER_UNARY_FUNC_IMPL(log10)
AD_PYOBJECT_ADAPTER_UNARY_FUNC_IMPL(sin)
AD_PYOBJECT_ADAPTER_UNARY_FUNC_IMPL(sinh)
AD_PYOBJECT_ADAPTER_UNARY_FUNC_IMPL(sqrt)
AD_PYOBJECT_ADAPTER_UNARY_FUNC_IMPL(tan)
AD_PYOBJECT_ADAPTER_UNARY_FUNC_IMPL(tanh)
AD_PYOBJECT_ADAPTER_UNARY_FUNC_IMPL(abs)
AD_PYOBJECT_ADAPTER_UNARY_FUNC_IMPL(sign)

#undef AD_PYOBJECT_ADAPTER_UNARY_FUNC_IMPL

std::ostream& operator<< (std::ostream &os, const galini::ad::ADPyobjectAdapter& x) {
  return os;
}

int Integer(const galini::ad::ADPyobjectAdapter& x) {
  return static_cast<int>(x);
}

#define AD_PYOBJECT_ADAPTER_COMPARE(name, op) \
  bool name(const galini::ad::ADPyobjectAdapter& x) \
  { return x op 0.0; }

AD_PYOBJECT_ADAPTER_COMPARE(GreaterThanZero, >)
AD_PYOBJECT_ADAPTER_COMPARE(GreaterThanOrZero, >=)
AD_PYOBJECT_ADAPTER_COMPARE(LessThanZero, <)
AD_PYOBJECT_ADAPTER_COMPARE(LessThanOrZero, <=)

#undef AD_PYOBJECT_ADAPTER_COMPARE

galini::ad::ADPyobjectAdapter CondExpOp(enum CompareOp,
					galini::ad::ADPyobjectAdapter&,
					galini::ad::ADPyobjectAdapter&,
					galini::ad::ADPyobjectAdapter&,
					galini::ad::ADPyobjectAdapter&) {
  throw std::runtime_error("Unsupported CondExpOP");
  return galini::ad::ADPyobjectAdapter(0.0);
}

} // namespace cppad
