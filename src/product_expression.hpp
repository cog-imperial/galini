#pragma once

#include <memory>

#include "binary_expression.hpp"

namespace galini {

template<typename T>
class ProductExpression : public BinaryExpression<T> {
public:
  using ptr = std::shared_ptr<ProductExpression<T>>;

  using BinaryExpression<T>::BinaryExpression;
};

}
