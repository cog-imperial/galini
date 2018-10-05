#pragma once

#include <memory>

#include "unary_function_expression.hpp"

namespace galini {

template<typename T>
class CosExpression : public UnaryFunctionExpression<T> {
public:
  using ptr = std::shared_ptr<CosExpression<T>>;

  using UnaryFunctionExpression<T>::UnaryFunctionExpression;
};

}
