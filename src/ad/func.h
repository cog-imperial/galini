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

#include <memory>
#include <pybind11/pybind11.h>

#include "ad/ad.h"

namespace py = pybind11;


namespace galini {

namespace ad {

template<class U>
class ADFunc {
public:
  explicit ADFunc(std::vector<AD<U>>&& x, std::vector<AD<U>>&& y) {
    f_ = std::make_unique<CppAD::ADFun<U>>(x, y);
  }

  ADFunc(ADFunc&& other) : f_(std::move(other.f_)) {}
  ADFunc(ADFunc& other) = delete;

  std::vector<U> forward(std::size_t q, const std::vector<U>& x) {
    return f_->Forward(q, x);
  }

  std::vector<U> reverse(std::size_t q, const std::vector<U>& w) {
    return f_->Reverse(q, w);
  }

  std::vector<U> hessian(const std::vector<U>& x, std::size_t l) {
    return f_->Hessian(x, l);
  }

  std::vector<U> hessian(const std::vector<U>& x, const std::vector<U>& w) {
    return f_->Hessian(x, w);
  }

private:
  std::unique_ptr<CppAD::ADFun<U>> f_;
};

template<>
class ADFunc<py::object> {
public:
  explicit ADFunc(std::vector<ADObject>&& x, std::vector<ADObject>&& y) {
    f_ = std::make_unique<CppAD::ADFun<ADPyobjectAdapter>>(x, y);
  }

  ADFunc(ADFunc&& other) : f_(std::move(other.f_)) {}
  ADFunc(ADFunc& other) = delete;

  std::vector<py::object> forward(std::size_t q, const std::vector<py::object>& x) {
    std::vector<ADPyobjectAdapter> pyx(x.size());
    std::copy(x.begin(), x.end(), pyx.begin());
    auto result = f_->Forward(q, pyx);
    return std::vector<py::object>(result.begin(), result.end());
  }

  std::vector<py::object> reverse(std::size_t q, const std::vector<py::object>& w) {
    std::vector<ADPyobjectAdapter> pyw(w.size());
    std::copy(w.begin(), w.end(), pyw.begin());
    auto result = f_->Reverse(q, pyw);
    return std::vector<py::object>(result.begin(), result.end());
  }

  std::vector<py::object> hessian(const std::vector<py::object>& x, std::size_t l) {
    std::vector<ADPyobjectAdapter> pyx(x.size());
    std::copy(x.begin(), x.end(), pyx.begin());
    auto result = f_->Hessian(pyx, l);
    return std::vector<py::object>(result.begin(), result.end());
  }

  std::vector<py::object> hessian(const std::vector<py::object>& x,
				  const std::vector<py::object>& w) {
    std::vector<ADPyobjectAdapter> pyx(x.size());
    std::copy(x.begin(), x.end(), pyx.begin());

    std::vector<ADPyobjectAdapter> pyw(w.size());
    std::copy(w.begin(), w.end(), pyw.begin());

    auto result = f_->Hessian(pyx, pyw);
    return std::vector<py::object>(result.begin(), result.end());
  }

private:
  std::unique_ptr<CppAD::ADFun<ADPyobjectAdapter>> f_;
};

} // namespace ad

} // namespace galini
