#pragma once

#include <memory>

#include "unary_function_expression.hpp"

namespace galini {

template<typename T>
class TanExpression : public UnaryFunctionExpression<T> {
public:
  using ptr = std::shared_ptr<TanExpression<T>>;

  using UnaryFunctionExpression<T>::UnaryFunctionExpression;
};

}
