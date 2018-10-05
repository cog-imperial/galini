#pragma once

#include <memory>

#include "expression_base.hpp"

namespace galini {

template<typename T>
class Constant : public Expression<T> {
public:
  static const index_t DEFAULT_DEPTH = 1;
  using ptr = std::shared_ptr<Constant<T>>;

  Constant<T>(const typename Problem<T>::ptr& problem, T value)
    : Expression<T>(problem, Constant<T>::DEFAULT_DEPTH)
    , value_(value) {}

  Constant<T>(T value) : Constant<T>(nullptr, value) {}

  index_t default_depth() const override {
    return Constant<T>::DEFAULT_DEPTH;
  }

  bool is_constant() const override {
    return true;
  }

  T value() const {
    return value_;
  }

  std::vector<typename Expression<T>::ptr> children() const override {
    return std::vector<typename Expression<T>::ptr>();
  }

  typename Expression<T>::ptr nth_children(index_t n) const override {
    return nullptr;
  }

private:
  T value_;
};

}
