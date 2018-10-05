#pragma once

#include <memory>

#include "unary_function_expression.hpp"

namespace galini {

template<typename T>
class SinExpression : public UnaryFunctionExpression<T> {
public:
  using ptr = std::shared_ptr<SinExpression<T>>;

  using UnaryFunctionExpression<T>::UnaryFunctionExpression;
};

}
