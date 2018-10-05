#pragma once

#include <memory>

#include "nary_expression.hpp"

namespace galini {

template<typename T>
class SumExpression : public NaryExpression<T> {
public:
  using ptr = std::shared_ptr<SumExpression<T>>;

  using NaryExpression<T>::NaryExpression;
};

}
