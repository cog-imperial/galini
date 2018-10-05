#pragma once

#include <memory>

#include "unary_expression.hpp"

namespace galini {

template<typename T>
class NegationExpression : public UnaryExpression<T> {
public:
  using ptr = std::shared_ptr<NegationExpression<T>>;

  using UnaryExpression<T>::UnaryExpression;
};

}
