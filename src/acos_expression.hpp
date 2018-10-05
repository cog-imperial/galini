#pragma once

#include <memory>

#include "unary_function_expression.hpp"

namespace galini {

template<typename T>
class AcosExpression : public UnaryFunctionExpression<T> {
public:
  using ptr = std::shared_ptr<AcosExpression<T>>;

  using UnaryFunctionExpression<T>::UnaryFunctionExpression;
};

}
