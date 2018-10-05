#pragma once

#include <memory>

#include "unary_function_expression.hpp"

namespace galini {

template<typename T>
class AsinExpression : public UnaryFunctionExpression<T> {
public:
  using ptr = std::shared_ptr<AsinExpression<T>>;

  using UnaryFunctionExpression<T>::UnaryFunctionExpression;
};

}
