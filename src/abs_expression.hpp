#pragma once

#include <memory>

#include "unary_function_expression.hpp"

namespace galini {

template<typename T>
class AbsExpression : public UnaryFunctionExpression<T> {
public:
  using ptr = std::shared_ptr<AbsExpression<T>>;

  using UnaryFunctionExpression<T>::UnaryFunctionExpression;
};

}
