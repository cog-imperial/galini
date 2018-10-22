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
#include <vector>

#include "expression_base.h"

namespace galini {

namespace expression {

class Constant : public Expression {
public:
  static const index_t DEFAULT_DEPTH = 1;
  using ptr = std::shared_ptr<Constant>;

  Constant(const std::shared_ptr<Problem>& problem, double value)
    : Expression(problem, Constant::DEFAULT_DEPTH)
    , value_(value) {}

  Constant(double value) : Constant(nullptr, value) {}

  index_t default_depth() const override {
    return Constant::DEFAULT_DEPTH;
  }

  bool is_constant() const override {
    return true;
  }

  double value() const {
    return value_;
  }

  std::vector<Expression::ptr> children() const override {
    return std::vector<Expression::ptr>();
  }

  typename Expression::ptr nth_children(index_t n) const override {
    return nullptr;
  }

private:
  double value_;
};


} // namespace expression

} // namespace galini
