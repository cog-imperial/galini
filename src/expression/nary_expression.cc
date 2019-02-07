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
#include "nary_expression.h"

#include "ad/ad.h"

namespace galini {

namespace expression {

namespace detail {

struct ExpressionCmp {
  bool operator()(const Expression::ptr& lhs, const Expression::ptr& rhs) {
    return lhs->uid() < rhs->uid();
  }
};

}

ADFloat SumExpression::eval(values_ptr<ADFloat>& values) const {
  return eval_sum(values);
}

ADObject SumExpression::eval(values_ptr<ADObject>& values) const {
  return eval_sum(values);
}


LinearExpression::LinearExpression(const std::shared_ptr<Problem>& problem,
				   const std::vector<typename Expression::ptr>& children,
				   const std::vector<double>& coefficients,
				   double constant)
  : NaryExpression(problem, children), constant_(constant) {
  if (coefficients.size() != children.size()) {
    throw std::runtime_error("children and coefficients must have the same size");
  }

  for (index_t i = 0; i < coefficients.size(); ++i) {
    auto var = children[i];
    coefficients_[var->uid()] = coefficients[i];
  }
}

LinearExpression::LinearExpression(const std::shared_ptr<Problem>& problem,
				   const std::vector<LinearExpression::ptr>& expressions)
  : NaryExpression(problem) {

  std::set<Expression::ptr, detail::ExpressionCmp> unique_children;

  constant_ = 0.0;
  for (const auto& expr : expressions) {
    constant_ += expr->constant_;
    for (const auto& var : expr->children_) {
      unique_children.insert(var);
      auto uid = var->uid();
      if (coefficients_.find(uid) != coefficients_.end()) {
	coefficients_[uid] += expr->coefficient(var);
      } else {
	coefficients_[uid] = expr->coefficient(var);
      }
    }
  }

  std::copy(unique_children.begin(), unique_children.end(), std::back_inserter(children_));
  num_children_ = children_.size();
}

double LinearExpression::coefficient(const std::shared_ptr<Expression>& var) const {
  return coefficients_.at(var->uid());
}


ADFloat LinearExpression::eval(values_ptr<ADFloat>& values) const {
  return eval_linear(values);
}

ADObject LinearExpression::eval(values_ptr<ADObject>& values) const {
  return eval_linear(values);
}

QuadraticExpression::QuadraticExpression(const std::shared_ptr<Problem>& problem,
					 const std::vector<typename Expression::ptr>& vars1,
					 const std::vector<typename Expression::ptr>& vars2,
					 const std::vector<double>& coefficients)
  : NaryExpression(problem) {
  if ((vars1.size() != vars2.size()) || (vars1.size() != coefficients.size())) {
    throw std::runtime_error("vars1, vars2 and coefficients must have the same size");
  }

  std::set<Expression::ptr, detail::ExpressionCmp> unique_children;

  for (index_t i = 0; i < vars1.size(); ++i) {
    auto var1 = vars1[i];
    auto idx1 = var1->uid();
    auto var2 = vars2[i];
    auto idx2 = var2->uid();
    auto coefficient = coefficients[i];

    unique_children.insert(var1);
    unique_children.insert(var2);

    if (idx1 < idx2) {
      auto term = BilinearTerm{var1, var2, coefficient};
      terms_[std::make_tuple(idx1, idx2)] = term;
    } else {
      auto term = BilinearTerm{var2, var1, coefficient};
      terms_[std::make_tuple(idx2, idx1)] = term;
    }
  }

  std::copy(unique_children.begin(), unique_children.end(), std::back_inserter(children_));
  num_children_ = children_.size();
}

QuadraticExpression::QuadraticExpression(const std::shared_ptr<Problem>& problem,
					 const std::vector<QuadraticExpression::ptr>& expressions)
  : NaryExpression(problem) {

  std::set<Expression::ptr, detail::ExpressionCmp> unique_children;

  for (const auto& expr : expressions) {
    for (const auto& t : expr->terms_) {
      unique_children.insert(t.second.var1);
      unique_children.insert(t.second.var2);

      if (terms_.find(t.first) != terms_.end()) {
	auto existing = terms_[t.first];
	auto new_coefficient = existing.coefficient + t.second.coefficient;
	auto term = BilinearTerm{existing.var1, existing.var2, new_coefficient};
	terms_[t.first] = term;
      } else {
	terms_.emplace(t);
      }
    }
  }

  std::copy(unique_children.begin(), unique_children.end(), std::back_inserter(children_));
  num_children_ = children_.size();
}

double QuadraticExpression::coefficient(const std::shared_ptr<Expression>& v1,
					const std::shared_ptr<Expression>& v2) const {
  auto idx1 = std::min(v1->uid(), v2->uid());
  auto idx2 = std::max(v1->uid(), v2->uid());
  auto idx = std::make_tuple(idx1, idx2);
  auto term = terms_.at(idx);
  return term.coefficient;
}

std::vector<BilinearTerm> QuadraticExpression::terms() const {
  std::vector<BilinearTerm> result;
  for (const auto& t : terms_) {
    result.push_back(t.second);
  }
  return result;
}


ADFloat QuadraticExpression::eval(values_ptr<ADFloat>& values) const {
  return eval_quadratic(values);
}

ADObject QuadraticExpression::eval(values_ptr<ADObject>& values) const {
  return eval_quadratic(values);
}

} // namespace expression

} // namespace galini
