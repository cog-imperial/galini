#pragma once

#include <memory>
#include <vector>

#include "expression_base.hpp"

namespace galini {

template<typename T>
class NaryExpression : public Expression<T> {
public:
  using ptr = std::shared_ptr<NaryExpression<T>>;

  NaryExpression(const std::shared_ptr<Problem<T>>& problem,
		  const std::vector<typename Expression<T>::ptr>& children)
    : Expression<T>(problem), children_(children) {
    this->num_children_ = children_.size();
  }

  NaryExpression(const std::vector<typename Expression<T>::ptr>& children)
    : NaryExpression(nullptr, children) {}

  std::vector<typename Expression<T>::ptr> children() const override {
    return std::vector<typename Expression<T>::ptr>(children_);
  }

  typename Expression<T>::ptr nth_children(index_t n) const override {
    return children_.at(n);
  }

private:
  std::vector<typename Expression<T>::ptr> children_;
};

}
