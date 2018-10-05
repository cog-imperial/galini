#pragma once

#include <memory>

#include "problem.hpp"
#include "expression_base.hpp"

namespace galini {

template<typename T>
class Variable : public Expression<T> {
public:
  static const index_t DEFAULT_DEPTH = 0;
  using ptr = std::shared_ptr<Variable<T>>;

  Variable<T>(const typename Problem<T>::ptr& problem)
    : Expression<T>(problem, Variable<T>::DEFAULT_DEPTH) {}

  Variable<T>() : Variable<T>(nullptr) {}

  index_t default_depth() const override {
    return Variable<T>::DEFAULT_DEPTH;
  }

  std::vector<typename Expression<T>::ptr> children() const override {
    return std::vector<typename Expression<T>::ptr>();
  }

  typename Expression<T>::ptr nth_children(index_t n) const override {
    return nullptr;
  }
};

}
