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

#define AD_PYOBJECT_ADAPTER_UNARY_FUNC(func) \
  ADPyobjectAdapter func() const;

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
  ADPyobjectAdapter(py::object inner)
    : inner_(inner) {}

  ADPyobjectAdapter(const ADPyobjectAdapter& other)
    : ADPyobjectAdapter(other.inner_) {}

  ~ADPyobjectAdapter() = default;

  operator py::object() const { return inner_; }
  operator int() const {
    return py::int_(inner_.attr("to_int")());
  }

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

  AD_PYOBJECT_ADAPTER_UNARY_FUNC(acos)
  AD_PYOBJECT_ADAPTER_UNARY_FUNC(asin)
  AD_PYOBJECT_ADAPTER_UNARY_FUNC(atan)
  AD_PYOBJECT_ADAPTER_UNARY_FUNC(cos)
  AD_PYOBJECT_ADAPTER_UNARY_FUNC(cosh)
  AD_PYOBJECT_ADAPTER_UNARY_FUNC(exp)
  AD_PYOBJECT_ADAPTER_UNARY_FUNC(fabs)
  AD_PYOBJECT_ADAPTER_UNARY_FUNC(log)
  AD_PYOBJECT_ADAPTER_UNARY_FUNC(log10)
  AD_PYOBJECT_ADAPTER_UNARY_FUNC(sin)
  AD_PYOBJECT_ADAPTER_UNARY_FUNC(sinh)
  AD_PYOBJECT_ADAPTER_UNARY_FUNC(sqrt)
  AD_PYOBJECT_ADAPTER_UNARY_FUNC(tan)
  AD_PYOBJECT_ADAPTER_UNARY_FUNC(tanh)
  AD_PYOBJECT_ADAPTER_UNARY_FUNC(abs)
  AD_PYOBJECT_ADAPTER_UNARY_FUNC(sign)

  py::object inner() const {
    return inner_;
  }

  bool is_zero() const;
  bool is_one() const;
  ADPyobjectAdapter pow(const ADPyobjectAdapter&) const;
private:
  py::object inner_;
};

#undef AD_PYOBJECT_ADAPTER_ASSIGN_OPERATOR
#undef AD_PYOBJECT_ADAPTER_BINARY_OPERATOR
#undef AD_PYOBJECT_ADAPTER_COMPARE_OPERATOR
#undef AD_PYOBJECT_ADAPTER_UNARY_FUNC

} // namespace ad

} // namespace galini

namespace CppAD {

bool IdenticalCon(const galini::ad::ADPyobjectAdapter& x);
bool IdenticalPar(const galini::ad::ADPyobjectAdapter& x);
bool IdenticalZero(const galini::ad::ADPyobjectAdapter& x);
bool IdenticalOne(const galini::ad::ADPyobjectAdapter& x);
bool IdenticalEqualCon(const galini::ad::ADPyobjectAdapter& x,
		       const galini::ad::ADPyobjectAdapter& y);
bool IdenticalEqualPar(const galini::ad::ADPyobjectAdapter& x,
		       const galini::ad::ADPyobjectAdapter& y);
galini::ad::ADPyobjectAdapter pow(const galini::ad::ADPyobjectAdapter& x,
				  const galini::ad::ADPyobjectAdapter& y);

#define AD_PYOBJECT_ADAPTER_UNARY_FUNC(func) \
  galini::ad::ADPyobjectAdapter func(const galini::ad::ADPyobjectAdapter& x);

AD_PYOBJECT_ADAPTER_UNARY_FUNC(acos)
AD_PYOBJECT_ADAPTER_UNARY_FUNC(asin)
AD_PYOBJECT_ADAPTER_UNARY_FUNC(atan)
AD_PYOBJECT_ADAPTER_UNARY_FUNC(cos)
AD_PYOBJECT_ADAPTER_UNARY_FUNC(cosh)
AD_PYOBJECT_ADAPTER_UNARY_FUNC(exp)
AD_PYOBJECT_ADAPTER_UNARY_FUNC(fabs)
AD_PYOBJECT_ADAPTER_UNARY_FUNC(log)
AD_PYOBJECT_ADAPTER_UNARY_FUNC(log10)
AD_PYOBJECT_ADAPTER_UNARY_FUNC(sin)
AD_PYOBJECT_ADAPTER_UNARY_FUNC(sinh)
AD_PYOBJECT_ADAPTER_UNARY_FUNC(sqrt)
AD_PYOBJECT_ADAPTER_UNARY_FUNC(tan)
AD_PYOBJECT_ADAPTER_UNARY_FUNC(tanh)
AD_PYOBJECT_ADAPTER_UNARY_FUNC(abs)
AD_PYOBJECT_ADAPTER_UNARY_FUNC(sign)

#undef AD_PYOBJECT_ADAPTER_UNARY_FUNC

std::ostream& operator<< (std::ostream &os, const galini::ad::ADPyobjectAdapter& x);

CPPAD_AZMUL(galini::ad::ADPyobjectAdapter)

int Integer(const galini::ad::ADPyobjectAdapter& x);

#define AD_PYOBJECT_ADAPTER_COMPARE(name, op) \
  bool name(const galini::ad::ADPyobjectAdapter& x);

AD_PYOBJECT_ADAPTER_COMPARE(GreaterThanZero, >)
AD_PYOBJECT_ADAPTER_COMPARE(GreaterThanOrZero, >=)
AD_PYOBJECT_ADAPTER_COMPARE(LessThanZero, <)
AD_PYOBJECT_ADAPTER_COMPARE(LessThanOrZero, <=)

#undef AD_PYOBJECT_ADAPTER_COMPARE

galini::ad::ADPyobjectAdapter CondExpOp(enum CompareOp,
					galini::ad::ADPyobjectAdapter&,
					galini::ad::ADPyobjectAdapter&,
					galini::ad::ADPyobjectAdapter&,
					galini::ad::ADPyobjectAdapter&);

} // namespace cppad
