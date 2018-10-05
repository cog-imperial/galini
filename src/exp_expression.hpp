#pragma once

#include <memory>

#include "unary_function_expression.hpp"

namespace galini {

template<typename T>
class ExpExpression : public UnaryFunctionExpression<T> {
public:
  using ptr = std::shared_ptr<ExpExpression<T>>;

  using UnaryFunctionExpression<T>::UnaryFunctionExpression;
};

}
