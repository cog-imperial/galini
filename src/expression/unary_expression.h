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

#include "expression_base.h"

namespace galini {

namespace expression {

class UnaryExpression : public Expression {
public:
  using ptr = std::shared_ptr<UnaryExpression>;

  UnaryExpression(const std::shared_ptr<Problem>& problem,
		  const std::vector<typename Expression::ptr>& children)
    : Expression(problem) {
    if (children.size() != 1) {
      throw std::runtime_error("children must have size 1");
    }
    this->num_children_ = 1;
    child_ = children[0];
  }

  UnaryExpression(const std::vector<typename Expression::ptr>& children)
    : UnaryExpression(nullptr, children) {}

  std::vector<typename Expression::ptr> children() const override {
    return std::vector<typename Expression::ptr>({child_});
  }

  typename Expression::ptr nth_children(index_t n) const override {
    if (n > 0) {
      throw std::out_of_range("UnaryExpression");
    }
    return child_;
  }
protected:
  Expression::ptr child_;
};


class NegationExpression : public UnaryExpression {
public:
  using ptr = std::shared_ptr<NegationExpression>;

  using UnaryExpression::UnaryExpression;

  ADFloat eval(values_ptr<ADFloat>& values) const override {
    return -(*values)[child_];
  }

  ADObject eval(values_ptr<ADObject>& values) const override {
    return -(*values)[child_];
  }
};


} // namespace expression

} // namespace galini
