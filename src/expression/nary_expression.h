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

#include <pybind11/numpy.h>

#include "expression_base.h"

namespace galini {

namespace expression {

class NaryExpression : public Expression {
public:
  using ptr = std::shared_ptr<NaryExpression>;

  NaryExpression(const std::shared_ptr<Problem>& problem,
		 const std::vector<Expression::ptr>& children)
    : Expression(problem), children_(children) {
    this->num_children_ = children_.size();
  }

  NaryExpression(const std::vector<Expression::ptr>& children)
    : NaryExpression(nullptr, children) {}

  std::vector<Expression::ptr> children() const override {
    return std::vector<Expression::ptr>(children_);
  }

  typename Expression::ptr nth_children(index_t n) const override {
    return children_.at(n);
  }

protected:
  std::vector<Expression::ptr> children_;
};

class SumExpression : public NaryExpression {
public:
  using ptr = std::shared_ptr<SumExpression>;

  using NaryExpression::NaryExpression;

  ADFloat eval(const std::vector<ADFloat>& values) const override;
  ADObject eval(const std::vector<ADObject>& values) const override;

private:
  template<class T>
  T eval_sum(const std::vector<T>& values) const {
    T result(0.0);
    for (auto child : children_) {
      result += values[child->idx()];
    }
    return result;
  }
};

class LinearExpression : public NaryExpression {
public:
  using ptr = std::shared_ptr<LinearExpression>;
  using coefficients_t = std::vector<double>;

  LinearExpression(const std::shared_ptr<Problem>& problem,
		   const std::vector<typename Expression::ptr>& children,
		   const coefficients_t& coefficients,
		   double constant)
    : NaryExpression(problem, children), coefficients_(coefficients), constant_(constant) {}

  LinearExpression(const std::vector<typename Expression::ptr>& children,
		   const coefficients_t& coefficients,
		   double constant)
    : LinearExpression(nullptr, children, coefficients, constant) {}

  pybind11::array coefficients() const {
    return pybind11::array(coefficients_.size(), coefficients_.data());
  }

  double constant() const {
    return constant_;
  }


  ADFloat eval(const std::vector<ADFloat>& values) const override;
  ADObject eval(const std::vector<ADObject>& values) const override;

private:
  template<class T>
  T eval_linear(const std::vector<T>& values) const {
    T result(constant_);
    for (index_t i = 0; i < children_.size(); ++i) {
      result += values[children_[i]->idx()] * coefficients_[i];
    }
    return result;
  }

  coefficients_t coefficients_;
  double constant_;
};

} // namespace expression

} // namespace galini
