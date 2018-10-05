#pragma once

#include <memory>

#include "unary_function_expression.hpp"

namespace galini {

template<typename T>
class SqrtExpression : public UnaryFunctionExpression<T> {
public:
  using ptr = std::shared_ptr<SqrtExpression<T>>;

  using UnaryFunctionExpression<T>::UnaryFunctionExpression;
};

}
