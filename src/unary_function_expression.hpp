#pragma once

#include <memory>

#include "unary_expression.hpp"

namespace galini {

template<typename T>
class UnaryFunctionExpression : public UnaryExpression<T> {
public:
  using ptr = std::shared_ptr<UnaryFunctionExpression<T>>;

  using UnaryExpression<T>::UnaryExpression;
};

}
