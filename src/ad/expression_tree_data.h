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

#include <vector>

#include "ad/ad.h"
#include "ad/func.h"
#include "expression/expression_base.h"

namespace galini {

namespace ad {

using Expression = expression::Expression;

class ExpressionTreeData {
public:
  explicit ExpressionTreeData(const std::vector<Expression::const_ptr>& vertices)
    : vertices_(vertices) {}

  explicit ExpressionTreeData(std::vector<Expression::const_ptr>&& vertices)
    : vertices_(vertices) {}

  ExpressionTreeData(ExpressionTreeData&& other)
    : vertices_(std::move(other.vertices_)) {}

  ExpressionTreeData(const ExpressionTreeData&) = delete;
  ~ExpressionTreeData() = default;

  std::vector<Expression::const_ptr> vertices() const {
    return vertices_;
  }

  template<class U, class B>
  ADFunc<U> eval(const std::vector<B>& x) const {
    std::vector<AD<B>> X(x.size());
    std::vector<AD<B>> Y(1);

    std::vector<AD<B>> values(vertices_.size());
    for (index_t i = 0; i < x.size(); ++i) {
      X[i] = AD<B>(x[i]);
    }

    CppAD::Independent(X);

    for (index_t i = 0; i < x.size(); ++i) {
      values[i] = X[i];
    }

    for (auto vertex : vertices_) {
      auto result = vertex->eval(values);
      values[vertex->idx()] = result;
    }

    Y[0] = values[vertices_.size() - 1];
    return ADFunc<U>(std::move(X), std::move(Y));
  }

  ADFunc<py::object> eval(const std::vector<py::object>& x) const {
    std::vector<ADPyobjectAdapter> pyx(x.size());
    std::copy(x.begin(), x.end(), pyx.begin());
    return eval<py::object, ADPyobjectAdapter>(pyx);
  }

private:
  std::vector<Expression::const_ptr> vertices_;
};

}  // namespace ad

} // namespace galini
