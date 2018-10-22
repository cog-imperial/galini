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
private:
  std::vector<Expression::const_ptr> vertices_;
};

}  // namespace ad

} // namespace galini
