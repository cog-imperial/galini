#pragma once

#include <memory>

#include "binary_expression.hpp"

namespace galini {

template<typename T>
class PowExpression : public BinaryExpression<T> {
public:
  using ptr = std::shared_ptr<PowExpression<T>>;

  using BinaryExpression<T>::BinaryExpression;
};

}
