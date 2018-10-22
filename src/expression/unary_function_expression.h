/* Copyright 2018 Francesco Ceccon

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
======================================================================== */
#pragma once

#include "unary_expression.h"

namespace galini {

namespace expression {

class UnaryFunctionExpression : public UnaryExpression {
public:
  using ptr = std::shared_ptr<UnaryFunctionExpression>;

  using UnaryExpression::UnaryExpression;
};

class AbsExpression : public UnaryFunctionExpression {
public:
  using ptr = std::shared_ptr<AbsExpression>;

  using UnaryFunctionExpression::UnaryFunctionExpression;
};

class SqrtExpression : public UnaryFunctionExpression {
public:
  using ptr = std::shared_ptr<SqrtExpression>;

  using UnaryFunctionExpression::UnaryFunctionExpression;
};

class ExpExpression : public UnaryFunctionExpression {
public:
  using ptr = std::shared_ptr<ExpExpression>;

  using UnaryFunctionExpression::UnaryFunctionExpression;
};

class LogExpression : public UnaryFunctionExpression {
public:
  using ptr = std::shared_ptr<LogExpression>;

  using UnaryFunctionExpression::UnaryFunctionExpression;
};

class SinExpression : public UnaryFunctionExpression {
public:
  using ptr = std::shared_ptr<SinExpression>;

  using UnaryFunctionExpression::UnaryFunctionExpression;
};

class CosExpression : public UnaryFunctionExpression {
public:
  using ptr = std::shared_ptr<CosExpression>;

  using UnaryFunctionExpression::UnaryFunctionExpression;
};

class TanExpression : public UnaryFunctionExpression {
public:
  using ptr = std::shared_ptr<TanExpression>;

  using UnaryFunctionExpression::UnaryFunctionExpression;
};

class AsinExpression : public UnaryFunctionExpression {
public:
  using ptr = std::shared_ptr<AsinExpression>;

  using UnaryFunctionExpression::UnaryFunctionExpression;
};

class AcosExpression : public UnaryFunctionExpression {
public:
  using ptr = std::shared_ptr<AcosExpression>;

  using UnaryFunctionExpression::UnaryFunctionExpression;
};

class AtanExpression : public UnaryFunctionExpression {
public:
  using ptr = std::shared_ptr<AtanExpression>;

  using UnaryFunctionExpression::UnaryFunctionExpression;
};


} // namespace expression

} // namespace galini
