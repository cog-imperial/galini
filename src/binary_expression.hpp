#pragma once

#include <memory>

#include "expression_base.hpp"

namespace galini {

template<typename T>
class BinaryExpression : public Expression<T> {
public:
  using ptr = std::shared_ptr<BinaryExpression<T>>;

  BinaryExpression(const std::shared_ptr<Problem<T>>& problem,
		  const std::vector<typename Expression<T>::ptr>& children)
    : Expression<T>(problem) {
    if (children.size() != 2) {
      throw std::runtime_error("children must have size 2");
    }
    first_ = children[0];
    second_ = children[1];
  }

  BinaryExpression(const std::vector<typename Expression<T>::ptr>& children)
    : BinaryExpression(nullptr, children) {}

  std::vector<typename Expression<T>::ptr> children() const override {
    return std::vector<typename Expression<T>::ptr>({first_, second_});
  }

  typename Expression<T>::ptr nth_children(index_t n) const override {
    if (n > 1) {
      throw std::out_of_range("BinaryExpression");
    }
    return (n == 0) ? first_ : second_;
  }
private:
  typename Expression<T>::ptr first_;
  typename Expression<T>::ptr second_;
};

}
