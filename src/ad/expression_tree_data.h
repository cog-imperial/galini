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
#include "ad/values.h"
#include "expression/expression_base.h"

namespace galini {

namespace ad {

enum class ExpressionTreeDataStorage {
  vector = 1,
  map = 2,
};

using Expression = expression::Expression;

class ExpressionTreeData {
public:
  using Storage = ExpressionTreeDataStorage;

  explicit ExpressionTreeData(const std::vector<Expression::const_ptr>& vertices, Storage storage)
    : vertices_(vertices), storage_(storage) {
    compute_num_variables();
  }

  explicit ExpressionTreeData(std::vector<Expression::const_ptr>&& vertices, Storage storage)
    : vertices_(vertices), storage_(storage) {
    compute_num_variables();
  }

  ExpressionTreeData(ExpressionTreeData&& other)
    : vertices_(std::move(other.vertices_)), storage_(other.storage_) {
    compute_num_variables();
  }

  ExpressionTreeData(const ExpressionTreeData&) = delete;
  ~ExpressionTreeData() = default;

  std::vector<Expression::const_ptr> vertices() const {
    return vertices_;
  }

  template<class U, class B>
  ADFunc<U> eval(const std::vector<B>& x, const std::vector<index_t>& out_indexes) const {
    std::vector<AD<B>> X(x.size());
    std::vector<AD<B>> Y(out_indexes.size());

    if (x.size() != num_variables_) {
      throw std::runtime_error("Invalid variables size, expected: " + std::to_string(num_variables_));
    }

    for (index_t i = 0; i < x.size(); ++i) {
      X[i] = AD<B>(x[i]);
    }

    CppAD::Independent(X);

    auto values = make_values<AD<B>>(vertices_.size());

    index_t variable_idx = 0;
    for (auto vertex : vertices_) {
      if (vertex->is_variable()) {
	(*values)[vertex] = X[variable_idx++];
      }

      // exit early to avoid iterations
      if (variable_idx >= num_variables_) {
	break;
      }
    }

    for (auto vertex : vertices_) {
      auto result = vertex->eval(values);
      (*values)[vertex] = result;
    }
    for (index_t i = 0; i < out_indexes.size(); ++i) {
      auto expr = vertices_[out_indexes[i]];
      Y[i] = (*values)[expr];
    }
    return ADFunc<U>(std::move(X), std::move(Y));
  }

  template<class U, class B>
  ADFunc<U> eval(const std::vector<B>& x) const {
    std::vector<index_t> out_indexes(1);
    out_indexes[0] = vertices_.size() - 1;
    return eval<U, B>(x, out_indexes);
  }


  ADFunc<py::object> eval(const std::vector<py::object>& x) const {
    std::vector<ADPyobjectAdapter> pyx(x.size());
    std::copy(x.begin(), x.end(), pyx.begin());
    return eval<py::object, ADPyobjectAdapter>(pyx);
  }

private:

  template<class AD>
  std::shared_ptr<Values<AD>> make_values(std::size_t size) const {
    if (storage_ == ExpressionTreeDataStorage::vector) {
      return std::make_shared<ValuesVector<AD>>(size);
    }
    return std::make_shared<ValuesMap<AD>>();
  }

  void compute_num_variables() {
    num_variables_ = 0;
    for (auto vertex : vertices_) {
      if (vertex->is_variable()) {
	num_variables_++;
      }
    }
  }

  std::vector<Expression::const_ptr> vertices_;
  Storage storage_;
  std::size_t num_variables_;
};

}  // namespace ad

} // namespace galini
