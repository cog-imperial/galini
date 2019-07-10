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

namespace detail {

struct IndexPairHash : public std::unary_function<std::tuple<index_t, index_t>, std::size_t> {
  std::size_t operator()(const std::tuple<index_t, index_t>& pair) const {
    auto h0 = std::hash<index_t>()(std::get<0>(pair));
    auto h1 = std::hash<index_t>()(std::get<1>(pair));
    return h0 ^ (h1 << 1);
  }
};

}

class NaryExpression : public Expression {
public:
  using ptr = std::shared_ptr<NaryExpression>;

  NaryExpression(const std::shared_ptr<Problem>& problem,
		 const std::vector<Expression::ptr>& children)
    : Expression(problem), children_(children) {
    this->num_children_ = children_.size();
  }
  NaryExpression(const std::shared_ptr<Problem>& problem)
    : Expression(problem) {
    children_ = {};
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

  ADFloat eval(values_ptr<ADFloat>& values) const override;
  ADObject eval(values_ptr<ADObject>& values) const override;

private:
  template<class T>
  T eval_sum(values_ptr<T>& values) const {
    T result(0.0);
    for (auto child : children_) {
      result += (*values)[child];
    }
    return result;
  }
};

class LinearExpression : public NaryExpression {
public:
  using ptr = std::shared_ptr<LinearExpression>;

  LinearExpression(const std::shared_ptr<Problem>& problem,
		   const std::vector<typename Expression::ptr>& children,
		   const std::vector<double>& coefficients,
		   double constant);

  LinearExpression(const std::vector<typename Expression::ptr>& children,
		   const std::vector<double>& coefficients,
		   double constant)
    : LinearExpression(nullptr, children, coefficients, constant) {}

  LinearExpression(const std::shared_ptr<Problem>& problem,
		   const std::vector<LinearExpression::ptr>& expressions);

  LinearExpression(const std::vector<LinearExpression::ptr>& expressions)
    : LinearExpression(nullptr, expressions) {}

  double coefficient(const std::shared_ptr<Expression>& var) const;

  double constant() const {
    return constant_;
  }

  std::vector<double> linear_coefs() const {
    auto size = children_.size();
    std::vector<double> coefs(size);
    for (index_t i = 0; i < size; ++i) {
      const auto var = children_[i];
      const auto coef = coefficients_.at(var->uid());
      coefs[i] = coef;
    }
    return coefs;
  }


  ADFloat eval(values_ptr<ADFloat>& values) const override;
  ADObject eval(values_ptr<ADObject>& values) const override;

private:
  template<class T>
  T eval_linear(values_ptr<T>& values) const {
    T result(constant_);
    for (index_t i = 0; i < children_.size(); ++i) {
      const auto var = children_[i];
      const auto coeff = coefficients_.at(var->uid());
      result += (*values)[var] * T(coeff);
    }
    return result;
  }

  std::unordered_map<index_t, double> coefficients_;
  double constant_;
};


struct BilinearTerm {
  std::shared_ptr<Expression> var1;
  std::shared_ptr<Expression> var2;
  double coefficient;
};


class QuadraticExpression : public NaryExpression {
public:
  using ptr = std::shared_ptr<QuadraticExpression>;

  QuadraticExpression(const std::shared_ptr<Problem>& problem,
		      const std::vector<typename Expression::ptr>& vars1,
		      const std::vector<typename Expression::ptr>& vars2,
		      const std::vector<double>& coefficients);

  QuadraticExpression(const std::vector<typename Expression::ptr>& vars1,
		      const std::vector<typename Expression::ptr>& vars2,
		      const std::vector<double>& coefficients)
    : QuadraticExpression(nullptr, vars1, vars2, coefficients) {}

  QuadraticExpression(const std::shared_ptr<Problem>& problem,
		      const std::vector<QuadraticExpression::ptr>& expressions);

  QuadraticExpression(const std::vector<QuadraticExpression::ptr>& expressions)
    : QuadraticExpression(nullptr, expressions) {}

  double coefficient(const std::shared_ptr<Expression>& v1, const std::shared_ptr<Expression>& v2) const;
  std::vector<BilinearTerm> terms() const;

  ADFloat eval(values_ptr<ADFloat>& values) const override;
  ADObject eval(values_ptr<ADObject>& values) const override;

private:
  template<class T>
  T eval_quadratic(values_ptr<T>& values) const {
    T result(0.0);
    for (const auto& p : terms_) {
      auto t = p.second;
      result += (*values)[t.var1] * (*values)[t.var2] * T(t.coefficient);
    }
    return result;
  }

  using index_pair = std::tuple<index_t, index_t>;
  std::unordered_map<index_pair, BilinearTerm, detail::IndexPairHash> terms_;
};

} // namespace expression

} // namespace galini
