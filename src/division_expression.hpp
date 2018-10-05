#pragma once

#include <memory>

#include "binary_expression.hpp"

namespace galini {

template<typename T>
class DivisionExpression : public BinaryExpression<T> {
public:
  using ptr = std::shared_ptr<DivisionExpression<T>>;

  using BinaryExpression<T>::BinaryExpression;
};

}
