#pragma once

#include <memory>

#include "unary_function_expression.hpp"

namespace galini {

template<typename T>
class LogExpression : public UnaryFunctionExpression<T> {
public:
  using ptr = std::shared_ptr<LogExpression<T>>;

  using UnaryFunctionExpression<T>::UnaryFunctionExpression;
};

}
