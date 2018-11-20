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

class BinaryExpression : public Expression {
public:
  using ptr = std::shared_ptr<BinaryExpression>;

  BinaryExpression(const std::shared_ptr<Problem>& problem,
		  const std::vector<typename Expression::ptr>& children)
    : Expression(problem) {
    if (children.size() != 2) {
      throw std::runtime_error("children must have size 2");
    }
    this->num_children_ = 2;
    first_ = children[0];
    second_ = children[1];
  }

  BinaryExpression(const std::vector<typename Expression::ptr>& children)
    : BinaryExpression(nullptr, children) {}

  std::vector<typename Expression::ptr> children() const override {
    return std::vector<typename Expression::ptr>({first_, second_});
  }

  typename Expression::ptr nth_children(index_t n) const override {
    if (n > 1) {
      throw std::out_of_range("BinaryExpression");
    }
    return (n == 0) ? first_ : second_;
  }

protected:
  Expression::ptr first_;
  Expression::ptr second_;
};

class ProductExpression : public BinaryExpression {
public:
  using ptr = std::shared_ptr<ProductExpression>;

  using BinaryExpression::BinaryExpression;

  ADFloat eval(values_ptr<ADFloat>& values) const override {
    return (*values)[first_] * (*values)[second_];
  }

  ADObject eval(values_ptr<ADObject>& values) const override {
    return (*values)[first_] * (*values)[second_];
  }

};

class DivisionExpression : public BinaryExpression {
public:
  using ptr = std::shared_ptr<DivisionExpression>;

  using BinaryExpression::BinaryExpression;

  ADFloat eval(values_ptr<ADFloat>& values) const override {
    return (*values)[first_] / (*values)[second_];
  }

  ADObject eval(values_ptr<ADObject>& values) const override {
    return (*values)[first_] / (*values)[second_];
  }

};

class PowExpression : public BinaryExpression {
public:
  using ptr = std::shared_ptr<PowExpression>;

  using BinaryExpression::BinaryExpression;

  ADFloat eval(values_ptr<ADFloat>& values) const override;
  ADObject eval(values_ptr<ADObject>& values) const override;

};

} // namespace expression

} // namespace galini
