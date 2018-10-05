#pragma once

#include <memory>

#include "unary_function_expression.hpp"

namespace galini {

template<typename T>
class AtanExpression : public UnaryFunctionExpression<T> {
public:
  using ptr = std::shared_ptr<AtanExpression<T>>;

  using UnaryFunctionExpression<T>::UnaryFunctionExpression;
};

}
