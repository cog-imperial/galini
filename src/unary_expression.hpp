#pragma once

#include <memory>

#include "expression_base.hpp"

namespace galini {

template<typename T>
class UnaryExpression : public Expression<T> {
public:
  using ptr = std::shared_ptr<UnaryExpression<T>>;

  UnaryExpression(const std::shared_ptr<Problem<T>>& problem,
		  const std::vector<typename Expression<T>::ptr>& children)
    : Expression<T>(problem) {
    if (children.size() != 1) {
      throw std::runtime_error("children must have size 1");
    }
    this->num_children_ = 1;
    child_ = children[0];
  }

  UnaryExpression(const std::vector<typename Expression<T>::ptr>& children)
    : UnaryExpression(nullptr, children) {}

  std::vector<typename Expression<T>::ptr> children() const override {
    return std::vector<typename Expression<T>::ptr>({child_});
  }

  typename Expression<T>::ptr nth_children(index_t n) const override {
    if (n > 0) {
      throw std::out_of_range("UnaryExpression");
    }
    return child_;
  }
private:
  typename Expression<T>::ptr child_;
};

}
